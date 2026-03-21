"""
predict.py
----------
Loads the saved model + scaler and returns a risk score + category
for a single patient's input dict.
"""

import joblib
import numpy as np
import os

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "models/best_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "models/scaler.pkl")

FEATURE_COLUMNS = [
    "Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate",
    "Headache", "Swelling", "Bleeding", "AbdominalPain", "ReducedFetalMovement"
]

RISK_LABELS = {0: "low risk", 1: "mid risk", 2: "high risk"}
RISK_THRESHOLDS = {"low risk": "Routine monitoring",
                   "mid risk": "Schedule teleconsultation",
                   "high risk": "Immediate doctor alert"}


def load_model():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_risk(patient: dict) -> dict:
    """
    Parameters
    ----------
    patient : dict with keys matching FEATURE_COLUMNS

    Returns
    -------
    dict with risk_score, risk_category, action, probabilities
    """
    model, scaler = load_model()

    # Build feature vector in correct column order
    features = np.array([[patient.get(col, 0) for col in FEATURE_COLUMNS]])
    features_scaled = scaler.transform(features)

    pred_class = int(model.predict(features_scaled)[0])
    proba      = model.predict_proba(features_scaled)[0].tolist()

    risk_score    = float(max(proba))
    risk_category = RISK_LABELS[pred_class]
    action        = RISK_THRESHOLDS[risk_category]

    return {
        "risk_score"    : round(risk_score, 4),
        "risk_category" : risk_category,
        "action"        : action,
        "probabilities" : {
            RISK_LABELS[i]: round(p, 4) for i, p in enumerate(proba)
        },
    }


if __name__ == "__main__":
    sample = {
        "Age": 28, "SystolicBP": 145, "DiastolicBP": 95,
        "BS": 16.5, "BodyTemp": 100.0, "HeartRate": 115,
        "Headache": 1, "Swelling": 1, "Bleeding": 0,
        "AbdominalPain": 1, "ReducedFetalMovement": 1,
    }
    result = predict_risk(sample)
    print("Risk Score    :", result["risk_score"])
    print("Risk Category :", result["risk_category"])
    print("Action        :", result["action"])
    print("Probabilities :", result["probabilities"])
