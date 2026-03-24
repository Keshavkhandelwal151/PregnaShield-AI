"""
evaluate.py
-----------
Standalone evaluation script.
Loads the saved model and scaler, runs evaluation on test data,
and prints a full report with accuracy, AUC-ROC, and confusion matrix.
 
Run with: python ml/evaluate.py
"""
 
import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
 
sys.path.append(os.path.dirname(__file__))
 
from preprocess import (
    load_uci_data,
    generate_synthetic_data,
    merge_datasets,
    preprocess,
    split_data,
    LABEL_MAP,
)
 
# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "models/best_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "models/scaler.pkl")
UCI_PATH    = os.path.join(os.path.dirname(__file__), "../data/maternal_health_risk.csv")
REPORT_DIR  = os.path.join(os.path.dirname(__file__), "models")
 
INV_LABEL   = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = [INV_LABEL[i] for i in sorted(INV_LABEL)]
N_CLASSES   = len(CLASS_NAMES)
 
 
# ── Load model ────────────────────────────────────────────────────────────────
 
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run ml/train_model.py first."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"  Model  loaded : {MODEL_PATH}")
    print(f"  Scaler loaded : {SCALER_PATH}")
    return model, scaler
 
 
# ── Load test data ────────────────────────────────────────────────────────────
 
def load_test_data(scaler):
    syn_df = generate_synthetic_data(n_samples=500)
    if os.path.exists(UCI_PATH):
        uci_df = load_uci_data(UCI_PATH)
        df = merge_datasets(uci_df, syn_df)
    else:
        print("  UCI CSV not found — evaluating on synthetic data only.")
        df = syn_df
 
    X, y, _ = preprocess(df, scaler=scaler, fit_scaler=False)
    _, X_test, _, y_test = split_data(X, y)
    print(f"  Test samples  : {len(y_test)}")
    return X_test, y_test
 
 
# ── Print classification report ───────────────────────────────────────────────
 
def print_report(y_test, y_pred, y_prob):
    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(
        y_test, y_prob, multi_class="ovr", average="macro"
    )
 
    print("\n" + "═" * 50)
    print("  EVALUATION REPORT")
    print("═" * 50)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUC-ROC   : {auc_score:.4f}")
    print("\n  Per-class breakdown:")
    print(
        classification_report(
            y_test, y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )
 
 
# ── Confusion matrix plot ─────────────────────────────────────────────────────
 
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "eval_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {path}")
 
 
# ── ROC curve plot (one-vs-rest) ──────────────────────────────────────────────
 
def plot_roc_curves(y_test, y_prob):
    y_bin = label_binarize(y_test, classes=list(range(N_CLASSES)))
    colours = ["#E53935", "#FDD835", "#43A047"]
 
    plt.figure(figsize=(7, 5))
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, colours)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=col, lw=2,
                 label=f"{cls} (AUC = {roc_auc:.2f})")
 
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — One vs Rest")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "eval_roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ROC curves saved      → {path}")
 
 
# ── Edge case tests ───────────────────────────────────────────────────────────
 
def run_edge_cases(model, scaler):
    """Predict on a few hand-crafted edge-case patients."""
    import pandas as pd
    from preprocess import FEATURE_COLUMNS
 
    edge_cases = [
        {
            "label"      : "Clearly HIGH risk",
            "Age": 38, "SystolicBP": 155, "DiastolicBP": 100,
            "BS": 18.0, "BodyTemp": 102.0, "HeartRate": 120,
            "Headache": 1, "Swelling": 1, "Bleeding": 1,
            "AbdominalPain": 1, "ReducedFetalMovement": 1,
        },
        {
            "label"      : "Clearly LOW risk",
            "Age": 22, "SystolicBP": 110, "DiastolicBP": 70,
            "BS": 7.5, "BodyTemp": 98.6, "HeartRate": 75,
            "Headache": 0, "Swelling": 0, "Bleeding": 0,
            "AbdominalPain": 0, "ReducedFetalMovement": 0,
        },
        {
            "label"      : "Borderline MID risk",
            "Age": 30, "SystolicBP": 132, "DiastolicBP": 88,
            "BS": 12.0, "BodyTemp": 99.5, "HeartRate": 95,
            "Headache": 1, "Swelling": 0, "Bleeding": 0,
            "AbdominalPain": 0, "ReducedFetalMovement": 0,
        },
    ]
 
    print("\n  Edge case predictions:")
    print("  " + "-" * 44)
    for case in edge_cases:
        label = case.pop("label")
        X = np.array([[case.get(c, 0) for c in FEATURE_COLUMNS]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]
        print(f"  {label}")
        print(f"    → Predicted : {INV_LABEL[pred]}")
        print(f"    → Confidence: {max(prob):.2f}")
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("\n── Loading artifacts ───────────────────────────")
    model, scaler = load_artifacts()
 
    print("\n── Loading test data ───────────────────────────")
    X_test, y_test = load_test_data(scaler)
 
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
 
    print_report(y_test, y_pred, y_prob)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curves(y_test, y_prob)
    run_edge_cases(model, scaler)
 
    print("\n✅ Evaluation complete.")
 
