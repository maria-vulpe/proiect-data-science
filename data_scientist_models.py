import os, numpy as np, pandas as pd, altair as alt, streamlit as st

from catboost import CatBoostClassifier

from data_scientist import CATBOOST_AVAILABLE

alt.themes.enable("dark"); alt.data_transformers.disable_max_rows()
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             classification_report, f1_score)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from model_utils import build_preprocessing_pipeline
from utils.label_maps import FRIENDLY_LABELS
import re
for k in ("trained", "yts", "ypred", "yprob",
          "rep_df", "classes", "imp_df"):
    st.session_state.setdefault(k, None)


def prettify_labels(series: pd.Series, target_name: str) -> pd.Series:
    mapping = FRIENDLY_LABELS.get(target_name, {})
    ser = series.replace(mapping)

    def _generic(x):
        if isinstance(x, (int, float)):
            return f"Score {x}"
        return re.sub(r"[_\\-]+", " ", str(x)).strip().title()

    return ser.apply(_generic)


def cm_altair(y_true,y_pred,cls,norm=False):
    cm = confusion_matrix(y_true, y_pred, normalize="true" if norm else None)
    df = (pd.DataFrame(cm, index=cls, columns=cls)
            .reset_index().melt("index", var_name="Predicted", value_name="Val")
            .rename(columns={"index":"Actual"}))
    ttl = "Confusion Matrix" + (" (normalized)" if norm else "")
    fmt = alt.Tooltip("Val:Q", format=".2%" if norm else "d")
    return (alt.Chart(df).mark_rect()
              .encode(x="Predicted:N", y="Actual:N",
                      color=alt.Color("Val:Q", scale=alt.Scale(scheme="blues")),
                      tooltip=["Actual","Predicted",fmt])
              .properties(title=ttl, width=320, height=320)
              .interactive())

def pr_altair(y_true,y_prob,cls):
    rows=[]
    if len(cls)==2:
        p,r,_ = precision_recall_curve(y_true, y_prob[:,1])
        rows.append(pd.DataFrame({"Recall":r,"Precision":p,"Class":cls[1]}))
    else:
        for i,c in enumerate(cls):
            p,r,_ = precision_recall_curve(y_true==i, y_prob[:,i])
            rows.append(pd.DataFrame({"Recall":r,"Precision":p,"Class":c}))
    data=pd.concat(rows, ignore_index=True)
    return (alt.Chart(data).mark_line(strokeWidth=2)
              .encode(x="Recall:Q", y="Precision:Q", color="Class:N",
                      tooltip=["Class","Recall","Precision"])
              .properties(title="Precision-Recall", width=360, height=360)
              .interactive())

def f1_altair(y_true,y_prob):
    thr=np.linspace(0,1,101)
    f1=[f1_score(y_true,(y_prob[:,1]>=t)) for t in thr]
    return (alt.Chart(pd.DataFrame({"Thr":thr,"F1":f1}))
              .mark_line(strokeWidth=2)
              .encode(x="Thr:Q", y="F1:Q", tooltip=["Thr","F1"])
              .properties(title="F1 vs Threshold", width=360, height=360)
              .interactive())

def proba_hist_altair(y_prob):
    df=pd.DataFrame({"Score":y_prob[:,1]})
    rule=alt.Chart(pd.DataFrame({"x":[0.5]})).mark_rule(color="red", strokeDash=[4,4])
    hist=alt.Chart(df).mark_bar().encode(
            x=alt.X("Score:Q", bin=alt.Bin(maxbins=40)),
            y="count()", tooltip=["count()"])
    return (hist+rule).properties(title="Probability Histogram", width=360, height=300).interactive()

def prf_altair(rep):
    melt=(rep.iloc[:-3][["precision","recall","f1-score"]]
          .reset_index().melt("index", var_name="Metric", value_name="Score"))
    return (alt.Chart(melt).mark_bar()
              .encode(x="index:N", y="Score:Q", color="Metric:N",
                      tooltip=["index","Metric","Score"])
              .properties(title="Precision / Recall / F1", width=320, height=300)
              .interactive())

def topfeat_altair(df_imp):
    zoom=alt.selection_interval(bind="scales")
    return (alt.Chart(df_imp).mark_bar()
              .encode(y=alt.Y("Feature:N", sort="-x"),
                      x="Importance:Q", tooltip=["Feature","Importance"],
                      color=alt.value("#ffbf66"))
              .add_params(zoom)
              .properties(title="Top Features", width=360, height=300))

def style_report(df):
    def hi(s): m=s.max(); return ["background:#314e6e;font-weight:600" if v==m else "" for v in s]
    return (df.style.format(precision=3)
            .apply(hi, axis=0)
            .set_table_styles([{"selector":"th",
                                "props":[("background","#2E3440"),("color","#E5E9F0")]}]))

# ───── Streamlit UI ───────────────────────────
def render():
    st.markdown("<h2 style='color:#8FBCBB;'>Model Evaluation Dashboard</h2>", unsafe_allow_html=True)

    # 1 ▸ Data loader
    with st.expander("  Select dataset & target", expanded=True):
        data_dir="data-gender"
        if not os.path.isdir(data_dir): st.error("Folder missing."); st.stop()
        files=[f for f in sorted(os.listdir(data_dir))
               if f.startswith("clean") and f.endswith(".csv")]
        if not files: st.error("No files."); st.stop()

        fname   = st.selectbox("Dataset", files)
        df      = pd.read_csv(os.path.join(data_dir, fname))
        df,_    = train_test_split(df, test_size=.8, random_state=42)
        st.dataframe(df.head(), height=160)

        tgt     = [c for c in df.columns if df[c].nunique() < 20]
        target  = st.selectbox("Target", [""]+tgt)
        defaults= [c for c in df.columns if c not in [target,"Patient_ID","Date"]][:6]
        feats   = st.multiselect("Features",
                                 [c for c in df.columns if c not in [target,"Patient_ID","Date"]],
                                 default=defaults)
    if not target or len(feats) < 2: st.stop()

    # 2. Pre-processing
    X = df[feats]
    y_raw = df[target]
    y_clean = prettify_labels(y_raw, target)

    y_enc, classes = pd.factorize(y_clean)

    pipe   = build_preprocessing_pipeline(X)
    X_proc = pipe.fit_transform(X)
    Xtr,Xts,ytr,yts = train_test_split(X_proc, y_enc,
                                       test_size=.3, random_state=42, stratify=y_enc)
    st.success(f"Train {Xtr.shape[0]} / Test {Xts.shape[0]} – {len(classes)} classes")

    # 3. Model training
    mdl = st.selectbox("Model", ["RandomForest","XGBoost","CatBoost"])
    if st.button("⚙️ Train & Evaluate"):
        if mdl == "RandomForest":
            model = RandomForestClassifier(n_estimators=200,
                                           class_weight="balanced",
                                           random_state=42)
        elif mdl == "XGBoost":
            model = XGBClassifier(n_estimators=200,
                                  learning_rate=.05,
                                  max_depth=6,
                                  subsample=.9,
                                  colsample_bytree=.9,
                                  eval_metric="logloss",
                                  use_label_encoder=False,
                                  random_state=42)
        elif mdl == "CatBoost" and CATBOOST_AVAILABLE:
            model = CatBoostClassifier(iterations=300,
                                       learning_rate=0.1,
                                       depth=6,
                                       auto_class_weights="Balanced",
                                       verbose=False,
                                       random_state=42)

        model.fit(Xtr, ytr)
        ypred = model.predict(Xts)
        yprob = model.predict_proba(Xts)
        rep   = pd.DataFrame(classification_report(yts, ypred,
                                                   target_names=classes,
                                                   output_dict=True)).T

        imp_df = None
        if hasattr(model, "feature_importances_"):
            try: fn = pipe.named_steps["preproc"].get_feature_names_out().tolist()
            except: fn = X.columns.tolist()
            imp_df = (pd.DataFrame({"Feature":fn, "Importance":model.feature_importances_})
                      .sort_values("Importance", ascending=False).head(15))

        # save & rerun
        st.session_state.update(
            trained=True,
            yts=yts,
            ypred=ypred,
            yprob=yprob,
            rep_df=rep,
            classes=classes,
            imp_df=imp_df
        )

        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    # 4. Display results
    if st.session_state.trained:
        yts     = st.session_state.yts
        ypred   = st.session_state.ypred
        yprob   = st.session_state.yprob
        rep     = st.session_state.rep_df
        classes = st.session_state.classes
        imp_df  = st.session_state.imp_df

        tabs = st.tabs(["Report", "Confusion", "PR Curve",
                        "F1 / Probabilities", "Distributions"])

        with tabs[0]:
            c1,c2 = st.columns([1.3,1])
            with c1: st.dataframe(style_report(rep), height=320)
            with c2: st.altair_chart(prf_altair(rep), width="stretch")
            if imp_df is not None:
                st.altair_chart(topfeat_altair(imp_df), width="stretch")

        with tabs[1]:
            st.altair_chart(cm_altair(yts, ypred, classes), width="stretch")
            st.altair_chart(cm_altair(yts, ypred, classes, norm=True), width="stretch")


        with tabs[2]:
            st.altair_chart(pr_altair(yts, yprob, classes), width="stretch")

        with tabs[3]:
            if len(classes) == 2:
                st.altair_chart(f1_altair(yts, yprob), width="stretch")
                st.altair_chart(proba_hist_altair(yprob), width="stretch")
            else:
                st.info("F1 & probability histogram only for binary problems.")

        with tabs[4]:
            num_cols = X.select_dtypes("number").columns
            if num_cols.empty:
                st.info("No nummeric collums.")
            else:
                col  = st.selectbox("Numeric variable", num_cols)
                bins = st.slider("Bin-uri", 10, 100, 30, 5)
                st.plotly_chart(
                    px.histogram(df, x=col, nbins=bins, color=target,
                                 barmode="overlay", title=f"Distribution {col}"),
                    width="stretch"
                )

if __name__=="__main__":
    render()