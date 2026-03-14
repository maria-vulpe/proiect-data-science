import streamlit as st
import pandas as pd

def render():
    st.markdown("<h1 style='text-align:center; color:#1ABC9C;'>HealthTrack</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#ECF0F1;'>A clinical intelligence dashboard for patient analysis and prediction.</p>",
        unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #1ABC9C;'>", unsafe_allow_html=True)

    st.markdown("""
    Welcome! Use the menu above to navigate:
    - **Clinician**: upload a model and predict hospitalization risk  
    - **Data Scientist**: explore/clean data, train models, cluster patients
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
          details { margin-bottom:1rem; }
          summary.feature-card {
            list-style:none; cursor:pointer;
            background:#273241; border-radius:12px; padding:1rem;
            box-shadow:0 4px 12px rgba(0,0,0,0.4);
            transition:transform .2s, box-shadow .2s;
            color:#ECF0F1;
          }
          summary.feature-card:hover {
            transform:translateY(-4px);
            box-shadow:0 8px 24px rgba(0,0,0,0.6);
            background:#1ABC9C; color:#1E1E2F;
          }
          summary.feature-card:hover h4,
          summary.feature-card:hover p {
            color:#1E1E2F !important;
          }
          details > div.preview {
            background:#1E1E2F;
            padding:0.75rem 1rem;
            border-radius:0 0 12px 12px;
            box-shadow:inset 0 4px 12px rgba(0,0,0,0.2);
            max-height:300px;
            overflow-y:auto;
            box-sizing:border-box;
          }

          .table-scroll {
            overflow-x:auto;
            -webkit-overflow-scrolling:touch;
          }
          .table-scroll .dataframe {
            white-space:nowrap;  /* fără wrap pe coloane */
          }

          .dataframe { border-collapse:collapse; }
          .dataframe th, .dataframe td { padding:4px 8px; color:#ECF0F1; }
          .dataframe th { background:#324152; }
          .dataframe td { background:#2C3E50; }

          .dataframe th { position:relative; }
          .dataframe th details { display:inline-block; }
          .dataframe th details summary {
            cursor:pointer; color:#1ABC9C; font-weight:500; list-style:none;
          }
          .dataframe th details summary:hover { text-decoration:underline; }
          .dataframe th details p {
            display:none; position:absolute; top:100%; left:0; z-index:20;
            width:240px; margin:0.25rem 0 0; padding:8px;
            background:#2C3E50; border:1px solid #1ABC9C; border-radius:6px;
            box-shadow:0 4px 12px rgba(0,0,0,0.5); color:#BDC3C7; font-size:0.85rem;
          }
          .dataframe th details[open] p { display:block; }

    div[role="option"] {
        user-select: none !important;
    }

    /* 2)  closed selectbox */
    .css-1wy0on6.egzxvld1 {  /* aceasta e clasa default Streamlit pentru selectbox */
        border: 1px solid #1ABC9C !important;
        background-color: #273241 !important;
        color: #ECF0F1 !important;
        border-radius: 6px !important;
        padding: 4px 8px !important;
    }

    /* 3)  opened list */
    div[role="listbox"] {
        background-color: #2C3E50 !important;
        border: 1px solid #1ABC9C !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
    }

    /* 4)  each option */
    div[role="option"] {
        padding: 8px 12px !important;
        font-size: 0.95rem;
        color: #ECF0F1 !important;
    }
    div[role="option"]:hover {
        background-color: #1ABC9C !important;
        color: #1E1E2F !important;
    }
     /* 1) Combobox închis */
      div[data-baseweb="select"] > div[role="combobox"] {
        border:1px solid #1ABC9C !important;
        background:#273241 !important;
        border-radius:6px !important;
        padding:0.25rem 0.5rem !important;
        color:#ECF0F1 !important;
      }
      /* 2) Listă deschisă */
      ul[role="listbox"] {
        background:#2C3E50 !important;
        border:1px solid #1ABC9C !important;
        border-radius:6px !important;
        box-shadow:0 4px 12px rgba(0,0,0,0.5) !important;
        margin-top:0 !important;
      }
      /* 3) Opțiuni */
      li[role="option"] {
        padding:0.5rem 1rem !important;
        color:#ECF0F1 !important;
        user-select:none !important;
      }
      li[role="option"]:hover,
      li[role="option"][aria-selected="true"] {
        background:#1ABC9C !important;
        color:#1E1E2F !important;
      }
       div[data-baseweb="select"] > div[role="combobox"] {
        border: 1px solid #1ABC9C !important;
        box-shadow: none !important;
      }
      div[data-baseweb="select"] > div[role="combobox"][aria-invalid="true"] {
        border: 1px solid #1ABC9C !important;
        box-shadow: none !important;
      }
      [data-testid="stSelectbox"] div[data-baseweb="select"] > div[role="combobox"] {
          border: 1px solid #1ABC9C !important;
      }
      [data-testid="stSelectbox"] div[data-baseweb="select"] > div[role="combobox"][aria-invalid="true"] {
          border-color: #1ABC9C !important;
      }
      [data-testid="stSelectbox"] div[data-baseweb="select"] > div[role="combobox"] *,
      [data-testid="stSelectbox"] div[data-baseweb="select"] > div[role="combobox"][aria-invalid="true"] * {
          box-shadow: none !important;
          outline: none    !important;
      }
      .css-1wy0on6.egzxvld1,
      .css-1wy0on6.egzxvld1[aria-invalid="true"] {
          border: 1px solid #1ABC9C !important;
          box-shadow: none !important;
      }
      details > summary {
        color: #1ABC9C !important;
      }


      details {
      margin-bottom: 1rem;
      border-radius: 8px;
      overflow: hidden;
      transition: box-shadow 0.3s ease;
      box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    }
    details:hover {
      box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    details[open] {
      box-shadow: 0 6px 16px rgba(0,0,0,0.5);
    }

    details > summary {
      position: relative;
      padding: 1rem 1.5rem;
      background: linear-gradient(90deg, #1E1E2F, #273241);
      color: #1ABC9C;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.3s ease, color 0.3s ease;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    details > summary::before {
      content: "";
      position: absolute;
      left: 0; top: 0;
      width: 4px; height: 100%;
      background: #1ABC9C;
      transform: scaleY(0);
      transform-origin: top;
      transition: transform 0.3s ease;
    }
    details[open] > summary::before {
      transform: scaleY(1);
    }

    details > summary::-webkit-details-marker {
      display: none;
    }
    details > summary::after {
      content: "⌄";
      font-size: 0.8rem;
      color: #BDC3C7;
      transition: transform 0.3s ease, color 0.3s ease;
    }
    details[open] > summary::after {
      transform: rotate(180deg);
      color: #ECF0F1;
    }

    details > div.preview {
      background: #1E1E2F;
      padding: 1rem;
      border-top: 3px solid #1ABC9C;
      box-sizing: border-box;
      overflow-x: auto;
      overflow-y: auto;
      white-space: nowrap;
    }

    .dataframe th, .dataframe td {
      padding: 8px 12px;
    }

    </style>
    """, unsafe_allow_html=True)

    # Dataset configurations
    datasets = [
        {
            "name": "clean_cardiopulmonary_data.csv",
            "desc": "Cardiopulmonary exercise test (CPET) results.",
            "path": "data/clean_cardiopulmonary_data.csv",
            "cols": {
                "Patient_ID": "Unique patient identifier across all files.",
                "Date": "Date when the CPET was performed.",
                "CPET": "Test protocol (e.g., 'Bruce protocol') and exercise tolerance notes.",
                "VO2max (mL/kg/min)": "Maximum oxygen consumption per kg body weight per minute.",
                "VE/VCO2 slope": "Ventilation-to-CO₂ production slope during exercise.",
                "METS": "Metabolic equivalent of task during exercise vs. rest.",
                "RER (Respiratory Exchange Ratio)": "Ratio of CO₂ produced to O₂ consumed."
            }
        },
        {
            "name": "clean_clinical_data.csv",
            "desc": "Vital signs and clinical observations.",
            "path": "data/clean_clinical_data.csv",
            "cols": {
                "Patient_ID": "Unique patient identifier.",
                "Date": "Date of clinical visit.",
                "Angina": "Presence of anginal chest pain (yes/no).",
                "Ascites": "Fluid accumulation in the abdomen (yes/no).",
                "Body Weight (Kg)": "Patient weight in kilograms.",
                "Systolic Blood Pressure (mmHg)": "Maximum arterial pressure during heart contraction.",
                "Diastolic Blood Pressure (mmHg)": "Minimum arterial pressure during heart relaxation.",
                "Dyspnea": "Shortness of breath (yes/no).",
                "Fatigue": "Unusual tiredness (yes/no).",
                "Heart Rate (bpm)": "Heartbeats per minute.",
                "Temperature (Celsius)": "Body temperature in degrees Celsius.",
                "Hepatomegaly": "Enlarged liver (yes/no).",
                "Jugular Venous Pressure": "Jugular vein pressure (<6 cm vs ≥6 cm).",
                "Peripheral Capillary Oxygen Saturation (%)": "O₂ saturation by pulse oximetry.",
                "Peripheral Hypoperfusion": "Signs of poor peripheral perfusion (yes/no).",
                "Peripheral Edema": "Peripheral swelling (yes/no).",
                "Pleural Effusion": "Fluid in the pleural space (yes/no).",
                "Pulmonary Rales": "Fine crackles on lung auscultation (yes/no).",
                "S3": "Presence of S3 heart sound (yes/no).",
                "S4": "Presence of S4 heart sound (yes/no).",
                "Syncope": "Sudden loss of consciousness (yes/no)."
            }
        },
        {
            "name": "clean_ecg_data.csv",
            "desc": "Electrocardiogram (ECG) parameters.",
            "path": "data/clean_ecg_data.csv",
            "cols": {
                "Patient_ID": "Patient identifier.",
                "Date": "ECG recording date.",
                "Conduction Delay": "Electrical conduction delay (yes/no).",
                "Sinus Rhythm": "Normal sinus rhythm (yes/no).",
                "Heart Rate (bpm)": "ECG-measured heart rate.",
                "PQ/PR (ms)": "Atrioventricular conduction interval.",
                "QRS (ms)": "Duration of the QRS complex.",
                "QT (ms)": "Ventricular depolarization and repolarization interval."
            }
        },
        {
            "name": "clean_echocardiography_data.csv",
            "desc": "Structural and functional echocardiography measurements.",
            "path": "data/clean_echocardiography_data.csv",
            "cols": {
                "Patient_ID": "Patient identifier.",
                "Date": "Echocardiography date.",
                "AR": "Aortic regurgitation (yes/no).",
                "AS": "Aortic stenosis (yes/no).",
                "Aortic Root diameter (mm)": "Aortic root diameter in millimeters.",
                "Basal right ventricle diameter (mm)": "Basal RV diameter in millimeters.",
                "Cardiac index (l/min/m²)": "Cardiac output adjusted for body surface area.",
                "EF (%)": "Left ventricular ejection fraction.",
                "Fractional area change (%)": "Right ventricular area change between diastole and systole.",
                "Inferior Vena Cava (IVC) (mm)": "IVC diameter in millimeters.",
                "Interventricular Septum Thickness (mm)": "Interventricular septum thickness.",
                "LVED Diameter (mm)": "Left ventricular end-diastolic diameter.",
                "LVES Diameter (mm)": "Left ventricular end-systolic diameter.",
                "Left Atrium Diameter (mm)": "Left atrium diameter.",
                "Left ventricular Posterior wall thickness (mm)": "LV posterior wall thickness.",
                "MR": "Mitral regurgitation (yes/no).",
                "MRAetiology": "Etiology of mitral regurgitation.",
                "MS": "Mitral stenosis (yes/no).",
                "RVSP (mmHg)": "Right ventricular systolic pressure.",
                "Right atrium area (cm²)": "Right atrial area.",
                "TAPSE (mm)": "Tricuspid annular plane systolic excursion.",
                "TR": "Tricuspid regurgitation (yes/no).",
                "TS": "Tricuspid stenosis (yes/no)."
            }
        },
        {
            "name": "clean_hospitalizations_data.csv",
            "desc": "Hospitalization records.",
            "path": "data/clean_hospitalizations_data.csv",
            "cols": {
                "Patient_ID": "Patient identifier.",
                "Start_Date": "Admission date.",
                "End_Date": "Discharge date.",
                "Unplanned": "Unplanned admission (yes/no).",
                "Retention_Event": "Significant clinical event during stay (yes/no).",
                "Event_Category": "Category of medical event."
            }
        },
        {
            "name": "clean_labs_data.csv",
            "desc": "Hematology and biochemistry results.",
            "path": "data/clean_labs_data.csv",
            "cols": {
                "Patient_ID": "Patient identifier.",
                "Date": "Lab measurement date.",
                "ALAT (U/L)": "Alanine aminotransferase (liver injury marker).",
                "ALB (g/dL)": "Serum albumin (protein synthesis marker).",
                "ALP (U/L)": "Alkaline phosphatase (bone and biliary marker).",
                "ASAT (U/L)": "Aspartate aminotransferase (liver and muscle).",
                "B12 (pg/mL)": "Vitamin B12 level.",
                "CPK (U/L)": "Total creatine kinase (muscle damage marker).",
                "CPK-MB (ng/mL)": "CK-MB isoenzyme (cardiac marker).",
                "CRP (mg/L)": "C-reactive protein (inflammation marker).",
                "Calcium (mg/dL)": "Serum calcium level.",
                "Cholesterol (mg/dL)": "Total cholesterol.",
                "Creatinine (mg/dL)": "Serum creatinine (kidney function).",
                "D-Dimer (ng/mL)": "Fibrin degradation product (thrombosis marker).",
                "Ferritin (ng/mL)": "Serum ferritin (iron stores).",
                "Fibrinogen (mg/dL)": "Fibrinogen level (coagulation marker).",
                "Glu (mg/dL)": "Blood glucose level.",
                "HDL (mg/dL)": "HDL cholesterol ('good').",
                "Haematocrit (%)": "Red blood cell percentage.",
                "Hemoglobin (g/dL)": "Hemoglobin concentration.",
                "INR": "International normalized ratio (anticoagulation monitoring).",
                "Iron (µg/dL)": "Serum iron level.",
                "LDH (U/L)": "Lactate dehydrogenase (tissue injury marker).",
                "LDL (mg/dL)": "LDL cholesterol ('bad').",
                "Magnesium (mg/dL)": "Serum magnesium level.",
                "Nt-proBNP (pg/mL)": "NT-proBNP (heart failure marker).",
                "PRCLT (%)": "Plateletcrit (platelet volume%).",
                "Phosphorus (mg/dL)": "Serum phosphorus level.",
                "Platelets (10³/µL)": "Platelet count.",
                "Potassium (mmol/L)": "Serum potassium level.",
                "Sodium (mmol/L)": "Serum sodium level.",
                "TSH (µIU/mL)": "Thyroid-stimulating hormone.",
                "Tbil (mg/dL)": "Total bilirubin.",
                "Tot prot (g/dL)": "Total serum protein.",
                "Triglycerides (mg/dL)": "Triglyceride level.",
                "Troponin (ng/mL)": "Troponin (cardiac injury).",
                "Urea (mg/dL)": "Blood urea nitrogen.",
                "Uric Acid (mg/dL)": "Serum uric acid.",
                "White Blood Cells (10³/µL)": "White blood cell count.",
                "fT3 (pg/mL)": "Free triiodothyronine.",
                "fT4 (ng/dL)": "Free thyroxine.",
                "γGT (U/L)": "Gamma-glutamyl transferase (biliary marker)."
            }
        },
        {
            "name": "clean_patients_health_status.csv",
            "desc": "Labels for overall patient health status.",
            "path": "data/clean_patients_health_status.csv",
            "cols": {
                "Patient_ID": "Patient identifier.",
                "Health_Status": "Overall health status (e.g., 'stable' vs. 'worsened')."
            }
        },
        {
            "name": "clean_visits_data.csv",
            "desc": "Clinical visit history.",
            "path": "data/clean_visits_data.csv",
            "cols": {
                "Patient_ID": "Patient identifier.",
                "Date": "Date of clinical visit.",
                "Type_of_Visit": "Type of service (consultation vs. admission).",
                "Planned": "Planned vs. unplanned visit (yes/no).",
                "Retention_Event": "Significant clinical event at visit (yes/no).",
                "Event_Category": "Medical event category (e.g., 'follow-up', 'acute exacerbation')."
            }
        }
    ]
    # Dataset configurations
    datasets = [
        {
            "name": "clean_cardiopulmonary_data.csv",
            "desc": "Cardiopulmonary exercise test (CPET) results.",
            "path": "data/clean_cardiopulmonary_data.csv",
            "cols": {
                "Patient_ID": "Unique patient identifier across all files.",
                "Date": "Date when the CPET was performed.",
                "CPET": "Test protocol (e.g., 'Bruce protocol') and exercise tolerance notes.",
                "VO2max (mL/kg/min)": "Maximum oxygen consumption per kg body weight per minute.",
                "VE/VCO2 slope": "Ventilation-to-CO₂ production slope during exercise.",
                "METS": "Metabolic equivalent of task during exercise vs. rest.",
                "RER (Respiratory Exchange Ratio)": "Ratio of CO₂ produced to O₂ consumed."
            }
        },
        {
            "name": "clean_clinical_data.csv",
            "desc": "Vital signs and clinical observations.",
            "path": "data/clean_clinical_data.csv",
            "cols": {
                "Patient_ID": "Unique patient identifier.",
                "Date": "Date of clinical visit.",
                "Angina": "Presence of anginal chest pain (yes/no).",
                "Ascites": "Fluid accumulation in the abdomen (yes/no).",
                "Body Weight (Kg)": "Patient weight in kilograms.",
                "Systolic Blood Pressure (mmHg)": "Maximum arterial pressure during heart contraction.",
                "Diastolic Blood Pressure (mmHg)": "Minimum arterial pressure during heart relaxation.",
                "Dyspnea": "Shortness of breath (yes/no).",
                "Fatigue": "Unusual tiredness (yes/no).",
                "Heart Rate (bpm)": "Heartbeats per minute.",
                "Temperature (Celsius)": "Body temperature in degrees Celsius.",
                "Hepatomegaly": "Enlarged liver (yes/no).",
                "Jugular Venous Pressure": "Jugular vein pressure (<6 cm vs ≥6 cm).",
                "Peripheral Capillary Oxygen Saturation (%)": "O₂ saturation by pulse oximetry.",
                "Peripheral Hypoperfusion": "Signs of poor peripheral perfusion (yes/no).",
                "Peripheral Edema": "Peripheral swelling (yes/no).",
                "Pleural Effusion": "Fluid in the pleural space (yes/no).",
                "Pulmonary Rales": "Fine crackles on lung auscultation (yes/no).",
                "S3": "Presence of S3 heart sound (yes/no).",
                "S4": "Presence of S4 heart sound (yes/no).",
                "Syncope": "Sudden loss of consciousness (yes/no)."
            }
        },
        {
            "name": "clean_ecg_data.csv",
            "desc": "Electrocardiogram (ECG) parameters.",
            "path": "data/clean_ecg_data.csv",
            "cols": {
                "Patient_ID": "Patient identifier.",
                "Date": "ECG recording date.",
                "Conduction Delay": "Electrical conduction delay (yes/no).",
                "Sinus Rhythm": "Normal sinus rhythm (yes/no).",
                "Heart Rate (bpm)": "ECG-measured heart rate.",
                "PQ/PR (ms)": "Atrioventricular conduction interval.",
                "QRS (ms)": "Duration of the QRS complex.",
                "QT (ms)": "Ventricular depolarization and repolarization interval."
            }
        },
        {
            "name": "clean_echocardiography_data.csv",
            "desc": "Structural and functional echocardiography measurements.",
            "path": "data/clean_echocardiography_data.csv",
            "cols": {
                "Patient_ID": "Patient identifier.",
                "Date": "Echocardiography date.",
                "AR": "Aortic regurgitation (yes/no).",
                "AS": "Aortic stenosis (yes/no).",
                "Aortic Root diameter (mm)": "Aortic root diameter in millimeters.",
                "Basal right ventricle diameter (mm)": "Basal RV diameter in millimeters.",
                "Cardiac index (l/min/m²)": "Cardiac output adjusted for body surface area.",
                "EF (%)": "Left ventricular ejection fraction.",
                "Fractional area change (%)": "Right ventricular area change between diastole and systole.",
                "Inferior Vena Cava (IVC) (mm)": "IVC diameter in millimeters.",
                "Interventricular Septum Thickness (mm)": "Interventricular septum thickness.",
                "LVED Diameter (mm)": "Left ventricular end-diastolic diameter.",
                "LVES Diameter (mm)": "Left ventricular end-systolic diameter.",
                "Left Atrium Diameter (mm)": "Left atrium diameter.",
                "Left ventricular Posterior wall thickness (mm)": "LV posterior wall thickness.",
                "MR": "Mitral regurgitation (yes/no).",
                "MRAetiology": "Etiology of mitral regurgitation.",
                "MS": "Mitral stenosis (yes/no).",
                "RVSP (mmHg)": "Right ventricular systolic pressure.",
                "Right atrium area (cm²)": "Right atrial area.",
                "TAPSE (mm)": "Tricuspid annular plane systolic excursion.",
                "TR": "Tricuspid regurgitation (yes/no).",
                "TS": "Tricuspid stenosis (yes/no)."
            }
        },
        {
            "name": "clean_hospitalizations_data.csv",
            "desc": "Hospitalization records.",
            "path": "data/clean_hospitalizations_data.csv",
            "cols": {
                "Patient_ID": "Patient identifier.",
                "Start_Date": "Admission date.",
                "End_Date": "Discharge date.",
                "Unplanned": "Unplanned admission (yes/no).",
                "Retention_Event": "Significant clinical event during stay (yes/no).",
                "Event_Category": "Category of medical event."
            }
        },
        {
            "name": "clean_labs_data.csv",
            "desc": "Hematology and biochemistry results.",
            "path": "data/clean_labs_data.csv",
            "cols": {
                "Patient_ID": "Patient identifier.",
                "Date": "Lab measurement date.",
                "ALAT (U/L)": "Alanine aminotransferase (liver injury marker).",
                "ALB (g/dL)": "Serum albumin (protein synthesis marker).",
                "ALP (U/L)": "Alkaline phosphatase (bone and biliary marker).",
                "ASAT (U/L)": "Aspartate aminotransferase (liver and muscle).",
                "B12 (pg/mL)": "Vitamin B12 level.",
                "CPK (U/L)": "Total creatine kinase (muscle damage marker).",
                "CPK-MB (ng/mL)": "CK-MB isoenzyme (cardiac marker).",
                "CRP (mg/L)": "C-reactive protein (inflammation marker).",
                "Calcium (mg/dL)": "Serum calcium level.",
                "Cholesterol (mg/dL)": "Total cholesterol.",
                "Creatinine (mg/dL)": "Serum creatinine (kidney function).",
                "D-Dimer (ng/mL)": "Fibrin degradation product (thrombosis marker).",
                "Ferritin (ng/mL)": "Serum ferritin (iron stores).",
                "Fibrinogen (mg/dL)": "Fibrinogen level (coagulation marker).",
                "Glu (mg/dL)": "Blood glucose level.",
                "HDL (mg/dL)": "HDL cholesterol ('good').",
                "Haematocrit (%)": "Red blood cell percentage.",
                "Hemoglobin (g/dL)": "Hemoglobin concentration.",
                "INR": "International normalized ratio (anticoagulation monitoring).",
                "Iron (µg/dL)": "Serum iron level.",
                "LDH (U/L)": "Lactate dehydrogenase (tissue injury marker).",
                "LDL (mg/dL)": "LDL cholesterol ('bad').",
                "Magnesium (mg/dL)": "Serum magnesium level.",
                "Nt-proBNP (pg/mL)": "NT-proBNP (heart failure marker).",
                "PRCLT (%)": "Plateletcrit (platelet volume%).",
                "Phosphorus (mg/dL)": "Serum phosphorus level.",
                "Platelets (10³/µL)": "Platelet count.",
                "Potassium (mmol/L)": "Serum potassium level.",
                "Sodium (mmol/L)": "Serum sodium level.",
                "TSH (µIU/mL)": "Thyroid-stimulating hormone.",
                "Tbil (mg/dL)": "Total bilirubin.",
                "Tot prot (g/dL)": "Total serum protein.",
                "Triglycerides (mg/dL)": "Triglyceride level.",
                "Troponin (ng/mL)": "Troponin (cardiac injury).",
                "Urea (mg/dL)": "Blood urea nitrogen.",
                "Uric Acid (mg/dL)": "Serum uric acid.",
                "White Blood Cells (10³/µL)": "White blood cell count.",
                "fT3 (pg/mL)": "Free triiodothyronine.",
                "fT4 (ng/dL)": "Free thyroxine.",
                "γGT (U/L)": "Gamma-glutamyl transferase (biliary marker)."
            }
        },
        {
            "name": "clean_patients_health_status.csv",
            "desc": "Labels for overall patient health status.",
            "path": "data/clean_patients_health_status.csv",
            "cols": {
                "Patient_ID": "Patient identifier.",
                "Health_Status": "Overall health status (e.g., 'stable' vs. 'worsened')."
            }
        },
        {
            "name": "clean_visits_data.csv",
            "desc": "Clinical visit history.",
            "path": "data/clean_visits_data.csv",
            "cols": {
                "Patient_ID": "Patient identifier.",
                "Date": "Date of clinical visit.",
                "Type_of_Visit": "Type of service (consultation vs. admission).",
                "Planned": "Planned vs. unplanned visit (yes/no).",
                "Retention_Event": "Significant clinical event at visit (yes/no).",
                "Event_Category": "Medical event category (e.g., 'follow-up', 'acute exacerbation')."
            }
        }
    ]

    for ds in datasets:
        try:
            preview_df = pd.read_csv(ds["path"], nrows=0)
            html_preview = preview_df.to_html(index=False, classes="dataframe", border=0)
            for column, description in ds["cols"].items():
                html_preview = html_preview.replace(
                    f"<th>{column}</th>",
                    f"<th><details><summary>{column}</summary><p>{description}</p></details></th>"
                )
        except Exception:
            html_preview = "<p style='color:#E74C3C;'>Preview unavailable</p>"

        with st.expander(f"{ds['name']} — {ds['desc']}"):
            placeholder = "-- Select a column --"
            column_list = [placeholder] + list(ds["cols"].keys())
            selected = st.selectbox(
                "Select a column for more details:",
                column_list,
                index=0,
                key=f"select_{ds['name']}"
            )

            if selected != placeholder:
                detail = ds["cols"][selected]
                st.markdown(
                    f"""
                       <div style="background:#1E2836; border-left:4px solid #1ABC9C; border-radius:6px;
                                   padding:1rem; margin:0.75rem 0; box-shadow:0 2px 8px rgba(0,0,0,0.5);">
                         <strong style="color:#1ABC9C; font-size:1.05rem;">{selected}</strong><br>
                         <span style="color:#BDC3C7; font-size:0.9rem;">{detail}</span>
                       </div>
                       """,
                    unsafe_allow_html=True
                )

            st.markdown("---")

            try:
                full_df = pd.read_csv(ds["path"])
                st.dataframe(full_df, height=400)
            except Exception as err:
                st.error(f"Unable to load full dataset: {err}")
