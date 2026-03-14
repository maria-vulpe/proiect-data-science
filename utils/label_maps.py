FRIENDLY_LABELS: dict[str, dict] = {
    "Gender": {
        "F": "Female",
        "M": "Male"
    },

    "CPET": {
        "I":   "NYHA I",
        "II":  "NYHA II",
        "III": "NYHA III",
        "IV":  "NYHA IV"
    },

    "Angina": {
        "no":           "None",
        "at exertion":  "At Exertion",
        "at rest":      "At Rest"
    },
    "Dyspnea": {
        "No":           "None",
        "at exertion":  "At Exertion",
        "bendopnea":    "Bendopnea"
    },
    "Fatigue": {
        "No":  "None",
        "Yes": "Present"
    },
    "Ascites": {
        "No":  "Absent",
        "Yes": "Present"
    },
    "Palpitations": {
        "No":  "No",
        "Yes": "Yes"
    },
    "Sinus Rhythm": {
        "No":  "No",
        "Yes": "Yes"
    },

    "AR": {"MILD": "Mild", "MODERATE": "Moderate", "SEVERE": "Severe"},
    "AS": {"MILD": "Mild", "MODERATE": "Moderate"},
    "MR": {"MILD": "Mild", "MODERATE": "Moderate", "SEVERE": "Severe"},
    "MS": {"MILD": "Mild", "MODERATE": "Moderate"},
    "TR": {"MILD": "Mild", "MODERATE": "Moderate", "SEVERE": "Severe"},
    "TS": {"MILD": "Mild", "MODERATE": "Moderate"},
    "MRAetiology": {
        "PRIMARY":   "Primary",
        "SECONDARY": "Secondary"
    },

    "Health_Status": {
        "HF":              "Heart Failure",
        "LVAD":            "LVAD",
        "Heart_Transplant":"Heart Transplant"
    },

    "Type_of_Visit": {
        "In person": "On-site",
        "Phonecall": "Phone"
    },
    "Event_Category": {
        "Heart failure related visit":                       "HF Visit",
        "HF decompensation needing iv. Diuretic-inotropes":  "HF Decomp +IV",
        "Device-related complications":                      "Device Comp.",
        "Infection management":                              "Infection",
        "Rejection episode":                                 "Rejection",
        "Unknown":                                           "Unknown"
    },
    "Retention_Event": {
        "True":    "Yes",
        "False":   "No",
        "Unknown": "Unknown"
    },

}