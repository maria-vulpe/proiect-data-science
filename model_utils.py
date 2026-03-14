# model_utils.py

import pandas as pd
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class TimeSeriesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, id_col: str = "Patient_ID", date_col: str = "Date"):
        self.id_col = id_col
        self.date_col = date_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df.loc[(df[col] < low) | (df[col] > high), col] = np.nan

        # 3) sort & interpolate per pacient
        if {self.id_col, self.date_col}.issubset(df.columns):
            df = df.sort_values([self.id_col, self.date_col])
            df[numeric_cols] = (
                df.groupby(self.id_col)[numeric_cols]
                .apply(lambda g: g.interpolate(method="linear"))
                .reset_index(level=0, drop=True)
            )

        return df


class DerivedFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if {"sbp", "dbp"}.issubset(X.columns):
            X["sbp_dbp_ratio"] = X["sbp"] / X["dbp"]

        return X


def build_preprocessing_pipeline(df: pd.DataFrame) -> Pipeline:

    import sklearn
    major, minor = map(int, sklearn.__version__.split(".")[:2])

    ts_preproc = TimeSeriesPreprocessor(id_col="Patient_ID", date_col="Date")
    temp_df = ts_preproc.fit_transform(df)

    num_cols = temp_df.select_dtypes(include="number").columns.tolist()
    cat_cols = temp_df.select_dtypes(include="object").columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    ohe_kwargs = dict(drop="first", handle_unknown="ignore")
    if (major, minor) >= (1, 4):
        ohe_kwargs["sparse_output"] = False     # scikit-learn 1.4+
    else:
        ohe_kwargs["sparse"] = False            # scikit-learn ≤1.3

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot",  OneHotEncoder(**ohe_kwargs)),
    ])

    # ColumnTransformer
    ct_kwargs = {}
    if (major, minor) >= (1, 3):
        ct_kwargs["verbose_feature_names_out"] = False

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
        **ct_kwargs,
    )

    full_pipeline = Pipeline([
        ("ts_preproc", ts_preproc),
        ("derive",     DerivedFeatures()),
        ("preproc",    preprocessor),
    ])

    return full_pipeline
