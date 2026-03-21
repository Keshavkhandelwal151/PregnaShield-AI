"""
train_model.py
--------------
Trains multiple ML models on the combined UCI + synthetic maternal health dataset.
Saves the best model and scaler to ml/models/.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)

from preprocess import (
    load_uci_data, generate_synthetic_data,
    merge_datasets, preprocess, split_data,
    LABEL_MAP
)

# ── Paths ─────────────────────────────────────────────────────────────────────
UCI_PATH    = "../data/maternal_health_risk.csv"
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Reverse label map for readable output ─────────────────────────────────────
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data():
    print("\n── Loading data ────────────────────────────────")
    syn_df = generate_synthetic_data(n_samples=500)
    print(f"  Synthetic data  : {len(syn_df)} rows")

    if os.path.exists(UCI_PATH):
        uci_df = load_uci_data(UCI_PATH)
        print(f"  UCI data        : {len(uci_df)} rows")
        df = merge_datasets(uci_df, syn_df)
    else:
        print("  UCI CSV not found — training on synthetic data only.")
        df = syn_df

    print(f"  Combined total  : {len(df)} rows")
    print("  Risk distribution:\n", df["RiskLevel"].value_counts().to_string())
    return df


# ── 2. Define models ──────────────────────────────────────────────────────────

def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1,
            max_depth=5, random_state=42
        ),
    }


# ── 3. Train & evaluate all models ───────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test):
    models  = get_models()
    results = {}

    print("\n── Training models ─────────────────────────────")
    for name, model in models.items():
        print(f"\n  ▸ {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)
            if hasattr(model, "predict_proba") else None
        )

        acc = accuracy_score(y_test, y_pred)
        auc = (
            roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
            if y_prob is not None else None
        )

        print(f"    Accuracy : {acc:.4f}")
        if auc:
            print(f"    AUC-ROC  : {auc:.4f}")
        print(classification_report(
            y_test, y_pred,
            target_names=[INV_LABEL[i] for i in sorted(INV_LABEL)],
            zero_division=0
        ))

        results[name] = {"model": model, "accuracy": acc, "auc": auc}

    return results


# ── 4. Pick best model ────────────────────────────────────────────────────────

def select_best(results: dict):
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    print(f"\n── Best model: {best_name} "
          f"(accuracy={results[best_name]['accuracy']:.4f}) ──")
    return best_name, results[best_name]["model"]


# ── 5. Confusion matrix plot ──────────────────────────────────────────────────

def plot_confusion(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = [INV_LABEL[i] for i in sorted(INV_LABEL)]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    print("  Confusion matrix saved → ml/models/confusion_matrix.png")
    plt.close()


# ── 6. Feature importance (Random Forest / GB only) ──────────────────────────

def plot_feature_importance(model, feature_names: list, model_name: str):
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in idx], rotation=45, ha="right")
    plt.title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=150)
    print("  Feature importance saved → ml/models/feature_importance.png")
    plt.close()


# ── 7. Save artifacts ─────────────────────────────────────────────────────────

def save_artifacts(model, scaler):
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n  Model  saved → {MODEL_PATH}")
    print(f"  Scaler saved → {SCALER_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocess import FEATURE_COLUMNS

    df = load_data()

    print("\n── Preprocessing ───────────────────────────────")
    X, y, scaler = preprocess(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"  Train : {X_train.shape}  Test : {X_test.shape}")

    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    best_name, best_model = select_best(results)

    plot_confusion(best_model, X_test, y_test, best_name)
    plot_feature_importance(best_model, FEATURE_COLUMNS, best_name)

    save_artifacts(best_model, scaler)
    print("\n✅ Training complete.")
