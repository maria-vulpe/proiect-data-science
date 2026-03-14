import pandas as pd
import numpy as np

file_paths = {
    "clinical_data": "./data/clinical_data.csv",
    "ecg_data": "./data/ecg_data.csv",
    "echocardiography_data": "./data/echocardiography_data.csv",
    "hospitalizations_data": "./data/hospitalizations_data.csv",
    "labs_data": "./data/labs_data.csv",
    "patients_health_status": "./data/patients_health_status.csv",
    "visits_data": "./data/visits_data.csv",
    "cardiopulmonary_data": "./data/cardiopulmonary_data.csv"
}

print(" Încărcăm fișierele CSV...")
data_frames = {name: pd.read_csv(path) for name, path in file_paths.items()}

for name, df in data_frames.items():
    print(f"\n Curățăm {name} - Dimensiune inițială: {df.shape}")
    df.drop_duplicates(inplace=True)
    print(f" Duplicați eliminați - Dimensiune nouă: {df.shape}")

# Tratăm valorile lipsă
for name, df in data_frames.items():
    for col in df.columns:
        if df[col].dtype == "object":  # Coloane textuale
            df[col].fillna("Unknown", inplace=True)
        elif df[col].dtype in ["int64", "float64"]:  # Coloane numerice
            df[col].fillna(df[col].median(), inplace=True)

#  Convertim coloanele de tip dată
for df_name in ["clinical_data", "ecg_data", "echocardiography_data", "labs_data", "visits_data", "cardiopulmonary_data"]:
    if "Date" in data_frames[df_name].columns:
        data_frames[df_name]["Date"] = pd.to_datetime(data_frames[df_name]["Date"], errors="coerce", dayfirst=True)

if "Start_Date" in data_frames["hospitalizations_data"].columns:
    data_frames["hospitalizations_data"]["Start_Date"] = pd.to_datetime(data_frames["hospitalizations_data"]["Start_Date"], errors="coerce", dayfirst=True)
if "End_Date" in data_frames["hospitalizations_data"].columns:
    data_frames["hospitalizations_data"]["End_Date"] = pd.to_datetime(data_frames["hospitalizations_data"]["End_Date"], errors="coerce", dayfirst=True)

#  Verificăm și raportăm dimensiunea finală a fiecărui dataset
def report_data_status():
    print("\n Raport final - dimensiuni dataset după curățare:")
    for name, df in data_frames.items():
        print(f"{name}: {df.shape}")

report_data_status()

# Exportăm datele curățate
for name, df in data_frames.items():
    df.to_csv(f"./data/clean_{name}.csv", index=False)
    print(f" {name} salvat în ./data/clean_{name}.csv")
