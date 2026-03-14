"""
Converts the two external UCI datasets into clean_*.csv files
compatible with the data-gender/ project schema.

Outputs:
  data-gender/clean_hf_clinical_records_uci.csv
      -> Labs + Echo features per patient, target = DEATH_EVENT
  data-gender/clean_hospitalizations_hf_uci.csv
      -> Hospitalization proxy derived from DEATH_EVENT
        (allows the 30-day hospitalization prediction feature)
  data-gender/clean_heart_statlog_uci.csv
      -> ECG / clinical features, target = HeartDisease (0/1)
"""

import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SRC_HFC = r"C:\Users\Mariah\Downloads\heart+failure+clinical+records\heart_failure_clinical_records_dataset.csv"
SRC_HFC_FALLBACK = r"C:\Users\Mariah\Downloads\heart_failure_clinical_records_dataset.csv"
SRC_STATLOG = r"C:\Users\Mariah\Downloads\dataset_53_heart-statlog.arff"

OUT_DIR = os.path.join(os.path.dirname(__file__), "data-gender")
EPOCH = datetime(2019, 1, 1)  # synthetic baseline date for all patients


# ──────────────────────────────────────────────────────────────────────────────
# 1. Heart Failure Clinical Records (UCI #519)
# ──────────────────────────────────────────────────────────────────────────────
def convert_hf_clinical():
    src = SRC_HFC if os.path.exists(SRC_HFC) else SRC_HFC_FALLBACK
    if not os.path.exists(src):
        print(f"[SKIP] Heart Failure Clinical Records not found at:\n  {SRC_HFC}\n  {SRC_HFC_FALLBACK}")
        return

    df = pd.read_csv(src)

    # ── patient metadata ──────────────────────────────────────────────────────
    n = len(df)
    df["Patient_ID"] = [f"HFC_{i+1:04d}" for i in range(n)]
    # `time` = follow-up period in days; Date = admission date = EPOCH - time
    df["Date"] = df["time"].apply(lambda t: (EPOCH - timedelta(days=int(t))).strftime("%Y-%m-%d"))

    # ── Gender (0=F, 1=M) ─────────────────────────────────────────────────────
    df["Gender"] = df["sex"].map({0: "F", 1: "M"})

    # ── rename columns to match project schema ────────────────────────────────
    df = df.rename(columns={
        "age":                     "Age",
        "ejection_fraction":       "EF (%)",
        "creatinine_phosphokinase": "CPK (U/L)",
        "serum_creatinine":        "Creatinine (mg/dL)",
        "serum_sodium":            "Sodium (Na) (mmol/L)",
        "diabetes":                "Diabetes",
        "anaemia":                 "Anaemia",
        "high_blood_pressure":     "Hypertension",
        "smoking":                 "Smoking",
        "time":                    "Follow_Up_Days",
        "DEATH_EVENT":             "DEATH_EVENT",
    })

    # platelets: original values are in cells/mL (e.g. 265000) ->
    # project unit is 10^3/uL  ≡  10^9/L  -> divide by 1000
    df["Platelets (10^3/uL)"] = (df["platelets"] / 1_000).round(1)

    # ── select & order columns ────────────────────────────────────────────────
    cols = [
        "Patient_ID", "Date", "Age", "Gender",
        "EF (%)", "CPK (U/L)", "Platelets (10^3/uL)",
        "Creatinine (mg/dL)", "Sodium (Na) (mmol/L)",
        "Diabetes", "Anaemia", "Hypertension", "Smoking",
        "Follow_Up_Days", "DEATH_EVENT",
    ]
    out_df = df[cols].copy()

    out_path = os.path.join(OUT_DIR, "clean_hf_clinical_records_uci.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(out_df)} rows -> {out_path}")

    # ── hospitalization proxy ─────────────────────────────────────────────────
    # For patients where DEATH_EVENT=1, create a hospitalization event that
    # ends on the follow-up date (EPOCH) and starts ~7 days earlier.
    # This lets the 30-day hospitalization prediction feature work.
    hosp_rows = []
    for _, row in df[df["DEATH_EVENT"] == 1].iterrows():
        end_dt = datetime.strptime(row["Date"], "%Y-%m-%d") + timedelta(days=int(row["Follow_Up_Days"]))
        start_dt = end_dt - timedelta(days=7)
        hosp_rows.append({
            "Patient_ID": row["Patient_ID"],
            "Start_Date": start_dt.strftime("%Y-%m-%d"),
            "End_Date":   end_dt.strftime("%Y-%m-%d"),
            "Unplanned": 1,
            "Retention_Event": 1,
            "Event_Category": "Mortality",
            "Gender": row["Gender"],
        })
    hosp_df = pd.DataFrame(hosp_rows)
    hosp_path = os.path.join(OUT_DIR, "clean_hospitalizations_hf_uci.csv")
    hosp_df.to_csv(hosp_path, index=False)
    print(f"[OK] Saved {len(hosp_df)} hospitalization proxy rows -> {hosp_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Heart Statlog (OpenML #53 / UCI Heart Disease Statlog)
# ──────────────────────────────────────────────────────────────────────────────
def parse_arff(path):
    """Minimal ARFF parser: returns (attribute_names, DataFrame)."""
    attrs = []
    data_lines = []
    in_data = False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@attribute"):
                # @attribute 'name' type
                m = re.match(r"@attribute\s+['\"]?([^'\"]+)['\"]?\s+(.+)", line, re.IGNORECASE)
                if m:
                    attrs.append(m.group(1).strip())
            elif line.lower().startswith("@data"):
                in_data = True
            elif in_data and line:
                data_lines.append(line.split(","))

    df = pd.DataFrame(data_lines, columns=attrs)
    # attempt numeric conversion for all columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            df[col] = df[col].str.strip()
    return df


CHEST_MAP = {1: "TA", 2: "ATA", 3: "NAP", 4: "ASY"}
ECG_MAP   = {0: "Normal", 1: "ST", 2: "LVH"}
SLOPE_MAP = {1: "Up", 2: "Flat", 3: "Down"}


def convert_statlog():
    if not os.path.exists(SRC_STATLOG):
        print(f"[SKIP] Statlog ARFF not found at {SRC_STATLOG}")
        return

    df = parse_arff(SRC_STATLOG)

    n = len(df)
    df["Patient_ID"] = [f"HTS_{i+1:04d}" for i in range(n)]
    # uniform synthetic dates, one per week
    df["Date"] = [(EPOCH + timedelta(weeks=i)).strftime("%Y-%m-%d") for i in range(n)]

    # Gender
    df["Gender"] = df["sex"].map({0: "F", 1: "M"})
    if df["Gender"].isna().any():
        # fallback: treat as numeric if already 0/1
        df["Gender"] = pd.to_numeric(df["sex"], errors="coerce").map({0: "F", 1: "M"})

    # HeartDisease: absent->0, present->1
    df["HeartDisease"] = df["class"].map({"absent": 0, "present": 1})

    # Categorical encodings -> readable strings (matching clean_heart_failure_data.csv)
    df["ChestPainType"] = pd.to_numeric(df["chest"], errors="coerce").map(CHEST_MAP).fillna(df["chest"])
    df["RestingECG"]    = pd.to_numeric(df["resting_electrocardiographic_results"], errors="coerce").map(ECG_MAP).fillna(df["resting_electrocardiographic_results"])
    df["ST_Slope"]      = pd.to_numeric(df["slope"], errors="coerce").map(SLOPE_MAP).fillna(df["slope"])
    df["ExerciseAngina"]= pd.to_numeric(df["exercise_induced_angina"], errors="coerce").map({0: "N", 1: "Y"}).fillna(df["exercise_induced_angina"])

    df = df.rename(columns={
        "age":                        "Age",
        "resting_blood_pressure":     "RestingBP",
        "serum_cholestoral":          "Cholesterol (mg/dL)",
        "fasting_blood_sugar":        "FastingBS",
        "maximum_heart_rate_achieved":"MaxHR",
        "oldpeak":                    "Oldpeak",
        "number_of_major_vessels":    "MajorVessels",
        "thal":                       "Thal",
    })

    cols = [
        "Patient_ID", "Date", "Age", "Gender",
        "ChestPainType", "RestingBP", "Cholesterol (mg/dL)",
        "FastingBS", "RestingECG", "MaxHR",
        "ExerciseAngina", "Oldpeak", "ST_Slope",
        "MajorVessels", "Thal", "HeartDisease",
    ]
    out_df = df[cols].copy()

    out_path = os.path.join(OUT_DIR, "clean_heart_statlog_uci.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(out_df)} rows -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Preparing external datasets ===")
    convert_hf_clinical()
    convert_statlog()
    print("=== Done ===")
