# File: clinician.py

import os
import io
import zipfile
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dotenv import load_dotenv
from utils.supa_io import list_artifacts, fetch_bytes

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm

load_dotenv()


def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    try:
        return model.predict(X, prediction_type="Probability").ravel()
    except Exception:
        margin = model.predict(X).ravel()
        return 1 / (1 + np.exp(-margin))


def render():
    st.markdown("<h2 style='color:#1ABC9C;'>Clinician Dashboard</h2>", unsafe_allow_html=True)
    st.write("Upload a pipeline & model (local sau cloud), apoi selectează un Patient_ID pentru predicție.")

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "model" not in st.session_state:
        st.session_state.model = None

    with st.expander("Upload Local Pipeline & Model"):
        pf = st.file_uploader("Pipeline (.pkl)", type="pkl")
        mf = st.file_uploader("Model    (.pkl)", type="pkl")
        if pf and mf:
            st.session_state.pipeline = pickle.load(pf)
            st.session_state.model = pickle.load(mf)
            st.success("✓ Loaded pipeline & model")

    with st.expander("Load Pipeline & Model from Cloud"):
        rows = list_artifacts(limit=500)
        models = [r for r in rows if r.get("type") == "model"]
        pipes = [r for r in rows if r.get("type") == "pipeline"]
        if models:
            labels = []
            for r in models:
                ts = r["created_at"][:16] if isinstance(r["created_at"], str) else str(r["created_at"])
                auc = r.get("auc")
                labels.append(
                    f"{ts} • {r['model_type']} • AUC {auc:.2f}"
                    if isinstance(auc, (int, float))
                    else f"{ts} • {r['model_type']} • AUC n/a"
                )
            sel = st.selectbox("Select model", labels)
            if st.button("Load selected"):
                rec = models[labels.index(sel)]
                sha = rec["git_sha"]
                try:
                    recp = next(p for p in pipes if p["git_sha"] == sha)
                except StopIteration:
                    st.error("No matching pipeline.")
                    return

                raw_m = fetch_bytes(rec["filename"])
                raw_p = fetch_bytes(recp["filename"])
                try:
                    st.session_state.model = pickle.loads(raw_m)
                    st.session_state.pipeline = pickle.loads(raw_p)
                    st.success("✓ Loaded from cloud")
                except Exception as e:
                    st.error(f"Could not deserialize: {e}")
                    return
        else:
            st.info("No models in cloud.")

    if st.session_state.pipeline is None or st.session_state.model is None:
        st.warning("Please load a pipeline & model first.")
        return

    data_dir = "data-gender"

    def read(fn, **kw):
        return pd.read_csv(os.path.join(data_dir, fn), **kw)

    clin    = read("clean_clinical_data_with_gender.csv",         parse_dates=["Date"])
    ecg     = read("clean_ecg_data_with_gender.csv",              parse_dates=["Date"])
    echo    = read("clean_echocardiography_data_with_gender.csv", parse_dates=["Date"])
    labs    = read("clean_labs_data_with_gender.csv",             parse_dates=["Date"])
    visits  = read("clean_visits_data_with_gender.csv",           parse_dates=["Date"])
    cp      = read("clean_cardiopulmonary_data_with_gender.csv",  parse_dates=["Date"])
    hosp    = read("clean_hospitalizations_data_with_gender.csv", parse_dates=["Start_Date", "End_Date"])
    patients_info = read("clean_patients_health_status_with_gender.csv")

    # ─── Select patient ───────────────────────────────────────────────────────
    pid = st.selectbox("Patient_ID", sorted(clin["Patient_ID"].unique()))
    st.write(f"Preparing data for **{pid}**…")

    # Demographics
    pr = patients_info[patients_info["Patient_ID"] == pid]
    if not pr.empty:
        gender = pr.iloc[0]["Gender"]
        raw_health = pr.iloc[0]["Health_Status"]
        mapping = {"HF": "Heart Failure", "HTN": "Hypertension", "DM": "Diabetes Mellitus"}
        health_stat = mapping.get(raw_health, raw_health)
    else:
        gender = health_stat = "N/A"

    # Merge temporal
    df = clin[clin["Patient_ID"] == pid].sort_values("Date").copy()
    sources = {"ecg": ecg, "echo": echo, "labs": labs, "visits": visits, "cp": cp, "hosp": hosp}
    for name, src in sources.items():
        sub = src[src["Patient_ID"] == pid]
        if sub.empty:
            continue
        if name == "hosp":
            sub = sub.sort_values("Start_Date")
            df = pd.merge_asof(
                df, sub,
                left_on="Date", right_on="Start_Date",
                by="Patient_ID", tolerance=pd.Timedelta("30d"),
                direction="backward", suffixes=("", "_hosp")
            )
            if "Start_Date" in df:
                df.rename(columns={"Start_Date": "Start_Date_hosp"}, inplace=True)
        else:
            sub = sub.sort_values("Date")
            df = pd.merge_asof(
                df, sub,
                on="Date", by="Patient_ID",
                tolerance=pd.Timedelta("30d"),
                direction="backward",
                suffixes=("", "_" + name)
            )

    df["Start_Date_hosp"] = df.get("Start_Date_hosp", pd.NaT)
    df["Hospitalization_Event"] = df["Start_Date_hosp"].notna().astype(int)

    st.dataframe(df.tail(5), height=200)

    # ─── Predict ──────────────────────────────────────────────────────────────
    def align_features(X_df, model):
        expected = None
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
        if expected is None:
            try:
                expected = model.get_booster().feature_names
            except Exception:
                pass
        if expected is None:
            raise ValueError("Model does not expose feature names.")
        for col in expected:
            if col not in X_df.columns:
                X_df[col] = 0.0
        return X_df[expected]

    with st.spinner("Predicting…"):
        Xp = st.session_state.pipeline.transform(df.copy())
        if not isinstance(Xp, pd.DataFrame):
            Xp = pd.DataFrame(Xp)

        try:
            proba = safe_predict_proba(st.session_state.model, Xp)
        except ValueError as ve:
            if "Feature shape mismatch" in str(ve):
                try:
                    Xp_aligned = align_features(Xp.copy(), st.session_state.model)
                    proba = safe_predict_proba(st.session_state.model, Xp_aligned)
                    st.warning(
                        f"Model-pipeline mismatch solved automatically: "
                        f"Pipeline produces {Xp.shape[1]} features, "
                        f"model expected {Xp_aligned.shape[1]}."
                    )
                    Xp = Xp_aligned
                except Exception as fix_err:
                    st.error(f"Error during automatic alignment: {fix_err}")
                    st.stop()
            else:
                st.error(f"Predict error: {ve}")
                st.stop()

        proba = np.clip(proba, 0, 1)
        st.info(f"Transformed shape: {Xp.shape}")

    df["Pred_Proba"] = proba

    # ─── Event flags ──────────────────────────────────────────────────────────
    vsub = visits[visits["Patient_ID"] == pid]

    clinic_unplanned_dates = set(vsub.loc[vsub["Planned"] == False, "Date"])
    iv_dates        = set(vsub.loc[vsub["Event_Category"].str.contains("iv",         case=False, na=False), "Date"])
    infection_dates = set(vsub.loc[vsub["Event_Category"].str.contains("infection",  case=False, na=False), "Date"])
    rejection_dates = set(vsub.loc[vsub["Event_Category"].str.contains("rejection",  case=False, na=False), "Date"])
    driveline_dates = set(vsub.loc[vsub["Event_Category"].str.contains("driveline",  case=False, na=False), "Date"])

    df["ClinicVisit_Event"]    = df["Date"].isin(clinic_unplanned_dates).astype(int)
    df["IV_Therapy_Event"]     = df["Date"].isin(iv_dates).astype(int)
    df["Infection_Event"]      = df["Date"].isin(infection_dates).astype(int)
    df["Rejection_Event"]      = df["Date"].isin(rejection_dates).astype(int)
    df["DrivelineInfect_Event"]= df["Date"].isin(driveline_dates).astype(int)

    event_styles = {
        "Hospitalization_Event":  {"label": "Hospitalisation",   "color": "#FF4136", "shape": "triangle-up"},
        "ClinicVisit_Event":      {"label": "Unplanned visit",    "color": "#FFA500", "shape": "circle"},
        "IV_Therapy_Event":       {"label": "IV therapy",         "color": "#2ECC40", "shape": "diamond"},
        "Infection_Event":        {"label": "Infection",          "color": "#0074D9", "shape": "square"},
        "Rejection_Event":        {"label": "Rejection (HT)",     "color": "#B10DC9", "shape": "triangle-down"},
        "DrivelineInfect_Event":  {"label": "Driveline infect",   "color": "#FF851B", "shape": "cross"},
    }

    # ─── Altair chart (interactive, displayed in app) ─────────────────────────
    base = (
        alt.Chart(df)
        .mark_line(point=False, strokeWidth=2, color="steelblue")
        .encode(
            x="Date:T",
            y=alt.Y(
                "Pred_Proba:Q",
                title="Predicted risk",
                scale=alt.Scale(domain=[
                    max(0, df.Pred_Proba.min() * 0.9),
                    df.Pred_Proba.max() * 1.1
                ])
            ),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Pred_Proba:Q", format=".2f")]
        )
        .properties(width=700, height=320)
    )

    layers = [base]
    for col, sty in event_styles.items():
        if col not in df.columns or df[col].sum() == 0:
            continue
        pts = (
            df.loc[df[col] == 1, ["Date", "Pred_Proba"]]
            .drop_duplicates("Date")
            .assign(EventType=sty["label"])
        )
        layers.append(
            alt.Chart(pts)
            .mark_point(size=110, filled=True,
                        shape=sty["shape"], color=sty["color"], opacity=0.9)
            .encode(
                x="Date:T",
                y="Pred_Proba:Q",
                tooltip=[
                    alt.Tooltip("Date:T"),
                    alt.Tooltip("Pred_Proba:Q", format=".2f"),
                    alt.Tooltip("EventType:N", title="Event")
                ]
            )
        )

    chart = (
        alt.layer(*layers)
        .interactive()
        .resolve_scale(color="independent", shape="independent")
        .configure_legend(orient="bottom")
    )

    # ── Use updated API: width='stretch' instead of width="stretch" ──
    st.altair_chart(chart, width="stretch")

    # ─── PNG export via matplotlib (no vl-convert needed) ────────────────────
    def build_png_bytes(data: pd.DataFrame) -> bytes:
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(data["Date"], data["Pred_Proba"], color="steelblue", linewidth=1.5, label="Predicted risk")

        marker_map = {
            "Hospitalization_Event":  ("^", "#FF4136"),
            "ClinicVisit_Event":      ("o", "#FFA500"),
            "IV_Therapy_Event":       ("D", "#2ECC40"),
            "Infection_Event":        ("s", "#0074D9"),
            "Rejection_Event":        ("v", "#B10DC9"),
            "DrivelineInfect_Event":  ("P", "#FF851B"),
        }
        legend_patches = [mpatches.Patch(color="steelblue", label="Predicted risk")]
        for col, (marker, color) in marker_map.items():
            if col not in data.columns or data[col].sum() == 0:
                continue
            pts = data.loc[data[col] == 1].drop_duplicates("Date")
            label = event_styles[col]["label"]
            ax.scatter(pts["Date"], pts["Pred_Proba"],
                       marker=marker, color=color, s=80, zorder=5, label=label)
            legend_patches.append(mpatches.Patch(color=color, label=label))

        ax.set_xlabel("Date", color="#ECEFF4")
        ax.set_ylabel("Predicted Risk", color="#ECEFF4")
        ax.set_title(f"Patient {pid} – Predicted Risk Over Time", color="#ECEFF4")
        ax.tick_params(colors="#ECEFF4")
        ax.set_facecolor("#2E3440")
        fig.patch.set_facecolor("#2E3440")
        ax.legend(handles=legend_patches, loc="upper left",
                  fontsize=7, facecolor="#3B4252", edgecolor="#4C566A", labelcolor="#ECEFF4")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    png_bytes = build_png_bytes(df)

    # ─── PDF ──────────────────────────────────────────────────────────────────
    hosp_dates = (
        df[df["Hospitalization_Event"] == 1]["Start_Date_hosp"]
        .dropna().dt.strftime("%Y-%m-%d").unique().tolist()
    )

    def build_pdf():
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf)
        styles = getSampleStyleSheet()
        elems = []

        elems.append(Paragraph(f"Patient Report – ID: {pid}", styles["Title"]))
        elems.append(Spacer(1, 0.5 * cm))
        elems.append(Paragraph(f"<b>Gender:</b> {gender}", styles["Normal"]))
        elems.append(Paragraph(f"<b>Health Status:</b> {health_stat}", styles["Normal"]))
        elems.append(Paragraph(f"<b>Generated:</b> {datetime.now():%Y-%m-%d %H:%M:%S}", styles["Normal"]))
        elems.append(Spacer(1, 1 * cm))
        elems.append(RLImage(io.BytesIO(png_bytes), width=17 * cm, height=9 * cm))
        elems.append(Spacer(1, 0.5 * cm))

        elems.append(Paragraph("Hospitalization Events:", styles["Heading3"]))
        if hosp_dates:
            table_data = [["Date of Hospitalization"]] + [[d] for d in hosp_dates]
            t = Table(table_data, colWidths=[10 * cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3B4252")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("TEXTCOLOR",  (0, 1), (-1, -1), colors.black),
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB"))
            ]))
            elems.append(t)
        else:
            elems.append(Paragraph("No hospitalization events detected.", styles["Normal"]))

        elems.append(Spacer(1, 0.5 * cm))
        elems.append(Paragraph("Notes:", styles["Heading3"]))
        elems.append(Paragraph("• <b>Blue line</b> = Model's predicted risk probability over time", styles["Normal"]))
        elems.append(Paragraph("  - Line connects predictions to show trend", styles["Normal"]))
        elems.append(Paragraph("• <b>Red triangles</b> = Confirmed hospitalization events", styles["Normal"]))
        elems.append(Paragraph("  - Shows when patient was actually hospitalized", styles["Normal"]))
        elems.append(Paragraph("  - Helps validate if model predicted correctly", styles["Normal"]))
        elems.append(Spacer(1, 0.3 * cm))
        elems.append(Paragraph("<b>Interpretation:</b>", styles["Normal"]))
        elems.append(Paragraph("• High probability (>0.5) before red marker = Good prediction ✓", styles["Normal"]))
        elems.append(Paragraph("• Low probability (<0.5) before red marker = Missed prediction ✗", styles["Normal"]))

        doc.build(elems)
        return buf.getvalue()

    pdf_bytes = build_pdf()

    # ─── ZIP download ─────────────────────────────────────────────────────────
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{pid}_report.pdf", pdf_bytes)
        zf.writestr(f"{pid}_chart.png", png_bytes)
    zip_io.seek(0)

    st.download_button(
        "Download Full Report (ZIP)",
        data=zip_io,
        file_name=f"patient_{pid}_report.zip",
        mime="application/zip"
    )


if __name__ == "__main__":
    render()