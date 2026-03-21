"""
preprocess.py
-------------
Handles data loading, cleaning, feature engineering, and train/test splitting
for both UCI Maternal Health Risk dataset and synthetic demo data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# ── Column names expected in the UCI dataset ──────────────────────────────────
UCI_COLUMNS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel"]

# ── Symptom columns added for the synthetic dataset ──────────────────────────
SYMPTOM_COLUMNS = [
    "Headache", "Swelling", "Bleeding", "AbdominalPain", "ReducedFetalMovement"
]

FEATURE_COLUMNS = [
    "Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"
] + SYMPTOM_COLUMNS

LABEL_COLUMN = "RiskLevel"
LABEL_MAP = {"low risk": 0, "mid risk": 1, "high risk": 2}


# ── 1. Load UCI dataset ───────────────────────────────────────────────────────

def load_uci_data(filepath: str) -> pd.DataFrame:
    """Load and lightly clean the UCI Maternal Health Risk CSV."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    # Standardise RiskLevel labels to lowercase
    df[LABEL_COLUMN] = df[LABEL_COLUMN].str.lower().str.strip()

    # Add missing symptom columns with 0 (UCI dataset has no symptom fields)
    for col in SYMPTOM_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    return df


# ── 2. Generate synthetic data ────────────────────────────────────────────────

def generate_synthetic_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic patient dataset with vital signs and symptoms.
    Rules are medically inspired (not clinically validated).
    """
    rng = np.random.default_rng(seed)

    age          = rng.integers(18, 45, n_samples)
    systolic_bp  = rng.integers(90, 160, n_samples)
    diastolic_bp = rng.integers(60, 110, n_samples)
    bs           = rng.uniform(6.0, 19.0, n_samples)          # mmol/L
    body_temp    = rng.uniform(98.0, 103.0, n_samples)        # °F
    heart_rate   = rng.integers(60, 130, n_samples)

    # Binary symptoms (1 = present)
    headache              = rng.integers(0, 2, n_samples)
    swelling              = rng.integers(0, 2, n_samples)
    bleeding              = rng.integers(0, 2, n_samples)
    abdominal_pain        = rng.integers(0, 2, n_samples)
    reduced_fetal_mvt     = rng.integers(0, 2, n_samples)

    # Derive risk label from heuristic rules
    risk = []
    for i in range(n_samples):
        score = 0
        if systolic_bp[i] >= 140 or diastolic_bp[i] >= 90:
            score += 2
        if bs[i] > 15:
            score += 2
        if body_temp[i] > 101:
            score += 1
        if heart_rate[i] > 110:
            score += 1
        score += bleeding[i] * 2 + reduced_fetal_mvt[i] * 2
        score += headache[i] + swelling[i] + abdominal_pain[i]

        if score >= 5:
            risk.append("high risk")
        elif score >= 2:
            risk.append("mid risk")
        else:
            risk.append("low risk")

    df = pd.DataFrame({
        "Age": age,
        "SystolicBP": systolic_bp,
        "DiastolicBP": diastolic_bp,
        "BS": bs,
        "BodyTemp": body_temp,
        "HeartRate": heart_rate,
        "Headache": headache,
        "Swelling": swelling,
        "Bleeding": bleeding,
        "AbdominalPain": abdominal_pain,
        "ReducedFetalMovement": reduced_fetal_mvt,
        "RiskLevel": risk,
    })
    return df


# ── 3. Merge datasets ─────────────────────────────────────────────────────────

def merge_datasets(uci_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate UCI and synthetic data, reset index."""
    combined = pd.concat([uci_df, synthetic_df], ignore_index=True)
    return combined


# ── 4. Preprocess features ────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, scaler: StandardScaler = None, fit_scaler: bool = True):
    """
    Encode labels, scale numeric features, return X, y, scaler.

    Parameters
    ----------
    df          : raw dataframe (merged or single source)
    scaler      : pass an existing fitted scaler when running inference
    fit_scaler  : True during training, False during prediction

    Returns
    -------
    X           : numpy array of features
    y           : numpy array of encoded labels
    scaler      : fitted StandardScaler (save this with the model)
    """
    df = df.copy()

    # Drop rows with missing target
    df = df.dropna(subset=[LABEL_COLUMN])

    # Encode label
    df["label"] = df[LABEL_COLUMN].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    y = df["label"].astype(int).values

    # Fill missing feature values with column median
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(df[col].median())

    X_raw = df[FEATURE_COLUMNS].values

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        X = scaler.transform(X_raw)

    return X, y, scaler


# ── 5. Train / test split ─────────────────────────────────────────────────────

def split_data(X, y, test_size: float = 0.2, seed: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    UCI_PATH = "data/maternal_health_risk.csv"

    print("Generating synthetic data …")
    syn_df = generate_synthetic_data(500)
    print(f"  Synthetic shape : {syn_df.shape}")

    if os.path.exists(UCI_PATH):
        print("Loading UCI data …")
        uci_df = load_uci_data(UCI_PATH)
        print(f"  UCI shape       : {uci_df.shape}")
        df = merge_datasets(uci_df, syn_df)
    else:
        print("UCI CSV not found — using synthetic only.")
        df = syn_df

    print(f"Combined shape  : {df.shape}")
    print("Risk distribution:\n", df["RiskLevel"].value_counts())

    X, y, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")
    print("Preprocessing complete ✓")
