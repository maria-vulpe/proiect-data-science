from datetime import datetime
import altair as alt

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from utils.supa_io import push_artifact

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
import xgboost as xgb

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from model_utils import build_preprocessing_pipeline

load_dotenv()


@st.cache_data(show_spinner=False)
def create_merged_training_dataset(data_dir: str) -> pd.DataFrame:
    """Merge all temporal CSVs exactly as Clinician does, for all patients."""
    def read(fn, **kw):
        return pd.read_csv(os.path.join(data_dir, fn), **kw)

    clin   = read("clean_clinical_data_with_gender.csv",         parse_dates=["Date"])
    ecg    = read("clean_ecg_data_with_gender.csv",              parse_dates=["Date"])
    echo   = read("clean_echocardiography_data_with_gender.csv", parse_dates=["Date"])
    labs   = read("clean_labs_data_with_gender.csv",             parse_dates=["Date"])
    visits = read("clean_visits_data_with_gender.csv",           parse_dates=["Date"])
    cp     = read("clean_cardiopulmonary_data_with_gender.csv",  parse_dates=["Date"])

    sources = {"ecg": ecg, "echo": echo, "labs": labs, "visits": visits, "cp": cp}
    all_patients = []
    for pid in clin["Patient_ID"].unique():
        df_p = clin[clin["Patient_ID"] == pid].sort_values("Date").copy()
        for name, src in sources.items():
            sub = src[src["Patient_ID"] == pid]
            if sub.empty:
                continue
            sub = sub.sort_values("Date")
            df_p = pd.merge_asof(
                df_p, sub,
                on="Date", by="Patient_ID",
                tolerance=pd.Timedelta("30d"),
                direction="backward",
                suffixes=("", "_" + name),
            )
        all_patients.append(df_p)
    return pd.concat(all_patients, ignore_index=True)


plt.style.use("dark_background")

sns.set_theme(
    context="talk",
    style="darkgrid",
    rc={
        "axes.facecolor": "#2E3440",
        "figure.facecolor": "#2E3440",
        "grid.color": "#4C566A",
        "text.color": "#ECEFF4",
        "axes.labelcolor": "#ECEFF4",
        "xtick.color": "#ECEFF4",
        "ytick.color": "#ECEFF4",
        "legend.facecolor": "#3B4252",
        "legend.edgecolor": "#4C566A",
        "axes.edgecolor": "#4C566A"
    }
)


def render():
    st.markdown("<h2 style='color:#88C0D0;'>Data Scientist Tools</h2>", unsafe_allow_html=True)

    # ─── 1. Clustering Analysis ───────────────────────────────────────────────
    with st.expander("1. Clustering Analysis of Patient Data", expanded=False):
        cluster_file = st.file_uploader(
            "Upload a dataset (CSV format)", type=["csv"], key="cluster_real"
        )
        df_cluster = None

        if cluster_file is not None:
            try:
                df_cluster = pd.read_csv(cluster_file)
                st.success("Dataset successfully loaded.")
                st.markdown("**Preview of the uploaded data:**")
                st.dataframe(df_cluster.head(), height=300)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                df_cluster = None

        if df_cluster is not None:
            numeric_cols = df_cluster.select_dtypes(include=np.number).columns.tolist()
            selected_features = st.multiselect(
                "Select numerical features for clustering analysis",
                options=numeric_cols,
                default=numeric_cols[: min(4, len(numeric_cols))],
                key="cluster_feats"
            )

            if len(selected_features) >= 2:
                X_cluster = df_cluster[selected_features].dropna()
                n_clusters = st.slider("Number of clusters (KMeans)", 2, 10, 3, key="cluster_k")

                if st.button("Run Clustering", key="cluster_run"):
                    try:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        cluster_labels = kmeans.fit_predict(X_cluster)

                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_cluster)

                        cluster_df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
                        cluster_df["Cluster"] = cluster_labels

                        fig, ax = plt.subplots(figsize=(4, 2.5))
                        sns.scatterplot(
                            data=cluster_df,
                            x="PCA1",
                            y="PCA2",
                            hue="Cluster",
                            palette="tab10",
                            ax=ax,
                            s=8,
                            edgecolor="w",
                            linewidth=0.2,
                        )
                        ax.set_title("PCA Projection of Patient Clusters", fontsize=12)
                        ax.tick_params(labelsize=8)
                        ax.legend(
                            title="Cluster", fontsize=7, title_fontsize=8, loc="upper right"
                        )
                        plt.tight_layout()
                        st.pyplot(fig)

                        df_cluster.loc[X_cluster.index, "Cluster"] = cluster_labels
                        st.markdown("**Cluster assignments added to the original dataset:**")
                        st.dataframe(df_cluster.head(10))
                    except Exception as e:
                        st.error(f"Error during clustering: {e}")
            else:
                st.warning("Select at least two numerical features for clustering.")

            st.markdown("--- Cluster Summary Statistics ---")
            if df_cluster is not None and "Cluster" in df_cluster.columns:
                cluster_summary = df_cluster.groupby("Cluster")[selected_features].mean().round(2)
                st.markdown("### Cluster Summary Statistics")
                st.dataframe(cluster_summary.style.highlight_max(axis=0), height=300)

                st.markdown("### Automatic Interpretation")
                for cluster_id, row in cluster_summary.iterrows():
                    top_feats = row.sort_values(ascending=False).head(2).index.tolist()
                    st.markdown(
                        f"- Cluster **{cluster_id}** has highest values in: "
                        f"**{top_feats[0]}**, **{top_feats[1]}**."
                    )
            else:
                st.warning("Clustering must be completed before statistical analysis.")

    # ─── 2. Model Trainer & Export Pickles ────────────────────────────────────
    with st.expander("2. Model Trainer", expanded=False):
        data_dir = "data-gender"
        if not os.path.isdir(data_dir):
            st.error(f"Folder '{data_dir}' does not exist. Please place clean_*.csv files there.")
        else:
            files = [f for f in os.listdir(data_dir) if f.startswith("clean") and f.endswith(".csv")]
            if not files:
                st.warning("No 'clean_*.csv' files found in the data directory.")
            else:
                use_merged = st.checkbox(
                    "🏥 Use merged dataset (recommended for hospitalization prediction — mirrors Clinician's data)",
                    key="use_merged_ds",
                )

                if use_merged:
                    st.info(
                        "Merging clinical + ECG + Echo + Labs + Visits + Cardiopulmonary for all patients "
                        "(same logic as Clinician). This ensures training features match inference features."
                    )
                    with st.spinner("Building merged dataset…"):
                        df = create_merged_training_dataset(data_dir)
                    dataset = "merged_all_sources"
                    st.success(f"Merged dataset ready — {df.shape[0]} rows × {df.shape[1]} columns")
                    st.dataframe(df.head(), height=200)
                else:
                    dataset = st.selectbox("Choose a dataset:", options=files, key="ds_trainer")
                    df = pd.read_csv(os.path.join(data_dir, dataset))
                    st.write(f"**{dataset}** — {df.shape[0]} rows × {df.shape[1]} columns")
                    st.dataframe(df.head(), height=200)

                all_cols = df.columns.tolist()

                # Optional: create hospitalization target
                if st.checkbox(
                    "Create hospitalization prediction target (30-day window)",
                    key="create_hosp_target"
                ):
                    if "Date" not in df.columns:
                        st.error("Dataset must have a 'Date' column for hospitalization prediction.")
                    else:
                        try:
                            hosp_file = "clean_hospitalizations_data_with_gender.csv"
                            hosp_path = os.path.join(data_dir, hosp_file)
                            if os.path.exists(hosp_path):
                                from datetime import timedelta

                                df["Date"] = pd.to_datetime(df["Date"])
                                hosp = pd.read_csv(
                                    hosp_path, parse_dates=["Start_Date", "End_Date"]
                                )

                                def will_be_hospitalized(row, hosp_data, days_ahead=30):
                                    patient_id = row["Patient_ID"]
                                    current_date = row["Date"]
                                    future_date = current_date + timedelta(days=days_ahead)
                                    patient_hosp = hosp_data[hosp_data["Patient_ID"] == patient_id]
                                    future_hosp = patient_hosp[
                                        (patient_hosp["Start_Date"] > current_date)
                                        & (patient_hosp["Start_Date"] <= future_date)
                                    ]
                                    return 1 if len(future_hosp) > 0 else 0

                                df["Hospitalization_Next_30d"] = df.apply(
                                    lambda row: will_be_hospitalized(row, hosp), axis=1
                                )

                                latest_date = df["Date"].max()
                                cutoff_date = latest_date - timedelta(days=30)
                                df = df[df["Date"] <= cutoff_date].copy()

                                st.success("Created 'Hospitalization_Next_30d' target column.")
                                hosp_counts = df["Hospitalization_Next_30d"].value_counts()
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("No hospitalization", f"{hosp_counts.get(0, 0):,}")
                                with col2:
                                    st.metric("Hospitalization", f"{hosp_counts.get(1, 0):,}")
                            else:
                                st.error(f"Hospitalizations file not found: {hosp_file}")
                        except Exception as e:
                            st.error(f"Error creating hospitalization target: {e}")

                all_cols = df.columns.tolist()
                target_opts = [c for c in all_cols if df[c].nunique() < 20]

                if "Hospitalization_Next_30d" in target_opts:
                    target_opts.remove("Hospitalization_Next_30d")
                    target_opts = ["Hospitalization_Next_30d 🏥"] + target_opts

                target = st.selectbox(
                    "Target variable (categorical)", [""] + target_opts, key="ds_target"
                )
                # strip display-only suffix (e.g. " 🏥") to get the real column name
                actual_target = target.replace(" 🏥", "") if target else target

                features = st.multiselect(
                    "Features (select at least 2)",
                    [c for c in all_cols if c not in [actual_target, "Patient_ID", "Date"]],
                    default=[
                        c for c in all_cols if c not in [actual_target, "Patient_ID", "Date"]
                    ][:5],
                    key="ds_feats"
                )

                if not target or len(features) < 2:
                    st.info("Select a target variable and at least two features.")
                else:
                    if df[actual_target].dtype in [np.int64, np.float64] and df[actual_target].nunique() > 10:
                        st.error(
                            "Selected target appears continuous. Please choose a categorical target."
                        )
                    else:
                        n_classes = df[actual_target].nunique()

                        model_options = ["RandomForest", "XGBoost"]
                        if CATBOOST_AVAILABLE:
                            model_options.append("CatBoost")
                        else:
                            st.warning(
                                "CatBoost not installed. Install with: `pip install catboost`"
                            )

                        model_choice = st.selectbox(
                            "Classification model", model_options, key="ds_model"
                        )

                        params = {}
                        if model_choice == "RandomForest":
                            params["n_estimators"] = st.slider(
                                "n_estimators (RF)", 50, 500, 100, key="rf_est"
                            )
                            params["max_depth"] = st.slider(
                                "max_depth (RF)", 3, 50, 10, key="rf_depth"
                            )
                        elif model_choice == "XGBoost":
                            params["n_estimators"] = st.slider(
                                "n_estimators (XGB)", 50, 500, 100, key="xgb_est"
                            )
                            params["learning_rate"] = st.slider(
                                "learning_rate (XGB)", 0.01, 0.5, 0.1, key="xgb_lr"
                            )
                        elif model_choice == "CatBoost":
                            params["iterations"] = st.slider(
                                "iterations (CatBoost)", 100, 1000, 300, key="cat_iter"
                            )
                            params["learning_rate"] = st.slider(
                                "learning_rate (CatBoost)", 0.01, 0.3, 0.1, key="cat_lr"
                            )
                            params["depth"] = st.slider(
                                "depth (CatBoost)", 3, 10, 6, key="cat_depth"
                            )

                        if st.button("Train & Export Pickles", key="ds_train"):
                            st.info(f"Starting training for {model_choice}...")

                            X_raw = df[features].copy()
                            y_raw = df[actual_target]
                            y = pd.factorize(y_raw)[0] if y_raw.dtype == object else y_raw

                            pipeline = build_preprocessing_pipeline(X_raw)
                            X_pre = pipeline.fit_transform(X_raw)

                            if model_choice == "RandomForest":
                                mdl = RandomForestClassifier(
                                    n_estimators=params["n_estimators"],
                                    max_depth=params["max_depth"],
                                    class_weight="balanced",
                                    random_state=42
                                )
                            elif model_choice == "XGBoost":
                                mdl = xgb.XGBClassifier(
                                    n_estimators=params["n_estimators"],
                                    learning_rate=params["learning_rate"],
                                    max_depth=6,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    use_label_encoder=False,
                                    eval_metric="logloss",
                                    random_state=42
                                )
                            elif model_choice == "CatBoost" and CATBOOST_AVAILABLE:
                                mdl = CatBoostClassifier(
                                    iterations=params["iterations"],
                                    learning_rate=params["learning_rate"],
                                    depth=params["depth"],
                                    auto_class_weights="Balanced",
                                    verbose=False,
                                    random_state=42
                                )

                            if n_classes == 2:
                                scoring = ["accuracy", "roc_auc", "f1"]
                            else:
                                scoring = ["accuracy", "f1_macro", "roc_auc_ovo"]

                            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                            cv_res = cross_validate(
                                mdl, X_pre, y, cv=cv,
                                scoring=scoring, return_train_score=False
                            )

                            mean_acc = np.mean(cv_res["test_accuracy"])
                            std_acc = np.std(cv_res["test_accuracy"])
                            st.write(
                                f"• **{model_choice}** Mean Test Accuracy: "
                                f"{mean_acc:.3f} ± {std_acc:.3f}"
                            )

                            mean_auc = None
                            if "test_roc_auc" in cv_res:
                                mean_auc = np.mean(cv_res["test_roc_auc"])
                                std_auc = np.std(cv_res["test_roc_auc"])
                                st.write(
                                    f"• **{model_choice}** Mean Test ROC AUC: "
                                    f"{mean_auc:.3f} ± {std_auc:.3f}"
                                )
                            elif "test_roc_auc_ovo" in cv_res:
                                mean_auc = np.mean(cv_res["test_roc_auc_ovo"])
                                std_auc = np.std(cv_res["test_roc_auc_ovo"])
                                st.write(
                                    f"• **{model_choice}** Mean Test ROC AUC (OVO): "
                                    f"{mean_auc:.3f} ± {std_auc:.3f}"
                                )

                            if "test_f1" in cv_res:
                                mean_f1 = np.mean(cv_res["test_f1"])
                                std_f1 = np.std(cv_res["test_f1"])
                                st.write(
                                    f"• **{model_choice}** Mean Test F1 Score: "
                                    f"{mean_f1:.3f} ± {std_f1:.3f}"
                                )
                            elif "test_f1_macro" in cv_res:
                                mean_f1 = np.mean(cv_res["test_f1_macro"])
                                std_f1 = np.std(cv_res["test_f1_macro"])
                                st.write(
                                    f"• **{model_choice}** Mean Test F1 Macro: "
                                    f"{mean_f1:.3f} ± {std_f1:.3f}"
                                )

                            # Feature importances
                            mdl.fit(X_pre, y)
                            if hasattr(mdl, "feature_importances_"):
                                importances = mdl.feature_importances_
                                try:
                                    feature_names = None
                                    if hasattr(pipeline, "get_feature_names_out"):
                                        feature_names = list(pipeline.get_feature_names_out())
                                    else:
                                        preproc = pipeline.named_steps.get("preproc", None)
                                        if preproc is not None and hasattr(
                                            preproc, "get_feature_names_out"
                                        ):
                                            feature_names = list(
                                                preproc.get_feature_names_out(
                                                    input_features=X_raw.columns
                                                )
                                            )
                                    if feature_names is None:
                                        raise RuntimeError("No feature names from pipeline")
                                except Exception:
                                    feature_names = list(X_raw.columns)

                                agg_dict = {col: 0.0 for col in X_raw.columns}
                                for fname, imp in zip(feature_names, importances):
                                    orig_col = fname.split("_")[0]
                                    if orig_col in agg_dict:
                                        agg_dict[orig_col] += float(imp)

                                importance_df = (
                                    pd.DataFrame(
                                        list(agg_dict.items()), columns=["feature", "importance"]
                                    )
                                    .assign(
                                        **{
                                            "importance (%)": lambda d: (
                                                d["importance"] * 100
                                            ).round(1)
                                        }
                                    )
                                    .loc[:, ["feature", "importance (%)"]]
                                    .sort_values("importance (%)", ascending=False)
                                    .head(10)
                                )
                                st.markdown("### Feature Importances")
                                st.dataframe(importance_df, height=300)

                            # Save locally
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            base_dataset = os.path.splitext(dataset)[0]

                            if actual_target == "Hospitalization_Next_30d":
                                model_filename = (
                                    f"model_hospitalization_{base_dataset}_{model_choice}_{ts}.pkl"
                                )
                                pipeline_filename = (
                                    f"pipeline_hospitalization_{base_dataset}_{model_choice}_{ts}.pkl"
                                )
                            else:
                                model_filename = f"model_{base_dataset}_{model_choice}_{ts}.pkl"
                                pipeline_filename = (
                                    f"pipeline_{base_dataset}_{model_choice}_{ts}.pkl"
                                )

                            model_path = os.path.join(data_dir, model_filename)
                            pipeline_path = os.path.join(data_dir, pipeline_filename)

                            with open(model_path, "wb") as f:
                                pickle.dump(mdl, f)
                            with open(pipeline_path, "wb") as f:
                                pickle.dump(pipeline, f)

                            model_meta = {
                                "model_type": model_choice,
                                "dataset": base_dataset,
                                "target": actual_target,
                                "git_sha": os.getenv("GIT_COMMIT", "dev"),
                                "auc": float(mean_auc) if mean_auc is not None else None,
                                "type": "model"
                            }
                            pipeline_meta = {
                                "model_type": model_choice,
                                "dataset": base_dataset,
                                "target": actual_target,
                                "git_sha": os.getenv("GIT_COMMIT", "dev"),
                                "type": "pipeline"
                            }

                            # Cloud upload
                            try:
                                art_model = push_artifact(model_path, model_meta)
                                art_pipeline = push_artifact(pipeline_path, pipeline_meta)
                                st.success("✅ Uploaded to cloud:")
                                st.write(f"• Model artifact: `{art_model}`")
                                st.write(f"• Pipeline artifact: `{art_pipeline}`")
                            except RuntimeError as err:
                                st.warning(str(err))
                                st.info(
                                    "Model and pipeline saved locally, but NOT uploaded to cloud."
                                )

                            st.success(
                                f"✅ {model_choice} model and pipeline pickles saved locally!"
                            )

                            if actual_target == "Hospitalization_Next_30d":
                                st.info(
                                    "This is a **hospitalization prediction model**. "
                                    "Use it in Clinician to see real 30-day hospitalization risk!"
                                )

                            st.markdown(
                                f"""
                                - **Model file**: `{model_filename}`
                                - **Pipeline file**: `{pipeline_filename}`
                                - **Dataset used**: `{dataset}`
                                - **Target**: `{target}`
                                - **Model type**: `{model_choice}`
                                - **Timestamp**: `{ts}`
                                - **Local paths**:
                                  ```
                                  {model_path}
                                  {pipeline_path}
                                  ```
                                """,
                                unsafe_allow_html=True
                            )

    # ─── 3. Gender-based Histograms ───────────────────────────────────────────
    with st.expander("3. Gender-based Histograms", expanded=False):
        gen_file = st.file_uploader(
            "Choose a CSV file that contains a 'Gender' column (F/M)",
            type=["csv"],
            key="gen_hist"
        )
        if gen_file is None:
            return

        try:
            df_gen = pd.read_csv(gen_file)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return

        if "Gender" not in df_gen.columns:
            st.error("The 'Gender' column is missing.")
            return

        numeric_cols = df_gen.select_dtypes(include=[np.number]).columns.tolist()
        df_clean = df_gen.dropna(subset=numeric_cols + ["Gender"])

        if not numeric_cols:
            st.warning("No numeric columns found.")
            return

        melted = df_clean.melt(
            id_vars=["Gender"],
            value_vars=numeric_cols,
            var_name="Variable",
            value_name="Value"
        )

        zoom = alt.selection_interval(bind="scales")

        base = (
            alt.Chart(melted)
            .mark_bar(opacity=0.6)
            .add_params(zoom)
            .encode(
                alt.X("Value:Q", bin=alt.Bin(maxbins=30), title=None),
                alt.Y("count()", title="Count"),
                alt.Color(
                    "Gender:N",
                    scale=alt.Scale(domain=["F", "M"], range=["#E91E63", "#2196F3"])
                )
            )
            .properties(width=300, height=150)
        )

        faceted = base.facet(
            row=alt.Row("Variable:N", header=alt.Header(labelAngle=0, title=None))
        ).resolve_scale(y="independent")

        st.altair_chart(faceted, width="stretch")


if __name__ == "__main__":
    render()