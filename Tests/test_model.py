"""
tests/test_model.py
--------------------
Tests for the trained ML model — accuracy checks, edge cases,
model persistence, and prediction consistency.
Run with: pytest tests/test_model.py -v
"""
 
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ml"))
 
import pytest
import numpy as np
import joblib
 
from preprocess import (
    generate_synthetic_data,
    preprocess,
    split_data,
    FEATURE_COLUMNS,
    LABEL_MAP,
)
 
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "../ml/models/best_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../ml/models/scaler.pkl")
 
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}
 
 
# ── Fixtures ──────────────────────────────────────────────────────────────────
 
@pytest.fixture(scope="module")
def trained_artifacts():
    """Load saved model and scaler. Skip all tests if not found."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model not found — run ml/train_model.py first.")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler
 
 
@pytest.fixture(scope="module")
def test_split(trained_artifacts):
    _, scaler = trained_artifacts
    df = generate_synthetic_data(n_samples=500, seed=99)
    X, y, _ = preprocess(df, scaler=scaler, fit_scaler=False)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_test, y_test
 
 
# ── Model persistence ─────────────────────────────────────────────────────────
 
class TestModelPersistence:
 
    def test_model_file_exists(self):
        assert os.path.exists(MODEL_PATH), \
            "Model file missing — run ml/train_model.py"
 
    def test_scaler_file_exists(self):
        assert os.path.exists(SCALER_PATH), \
            "Scaler file missing — run ml/train_model.py"
 
    def test_model_loads_without_error(self, trained_artifacts):
        model, _ = trained_artifacts
        assert model is not None
 
    def test_scaler_is_fitted(self, trained_artifacts):
        _, scaler = trained_artifacts
        assert hasattr(scaler, "mean_"), "Scaler is not fitted"
        assert len(scaler.mean_) == len(FEATURE_COLUMNS)
 
    def test_model_has_predict_method(self, trained_artifacts):
        model, _ = trained_artifacts
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
 
 
# ── Model accuracy ────────────────────────────────────────────────────────────
 
class TestModelAccuracy:
 
    def test_accuracy_above_threshold(self, trained_artifacts, test_split):
        from sklearn.metrics import accuracy_score
        model, _ = trained_artifacts
        X_test, y_test = test_split
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        assert acc >= 0.70, f"Accuracy too low: {acc:.2f} (threshold: 0.70)"
 
    def test_auc_above_threshold(self, trained_artifacts, test_split):
        from sklearn.metrics import roc_auc_score
        model, _ = trained_artifacts
        X_test, y_test = test_split
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        assert auc >= 0.75, f"AUC too low: {auc:.2f} (threshold: 0.75)"
 
    def test_predicts_all_three_classes(self, trained_artifacts, test_split):
        model, _ = trained_artifacts
        X_test, _ = test_split
        preds = set(model.predict(X_test))
        assert preds == {0, 1, 2}, \
            f"Model only predicts classes: {preds} — expected {{0, 1, 2}}"
 
    def test_probabilities_sum_to_one(self, trained_artifacts, test_split):
        model, _ = trained_artifacts
        X_test, _ = test_split
        probs = model.predict_proba(X_test)
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            "Probabilities do not sum to 1"
 
 
# ── Prediction consistency ────────────────────────────────────────────────────
 
class TestPredictionConsistency:
 
    def test_same_input_same_output(self, trained_artifacts):
        model, scaler = trained_artifacts
        patient = np.array([[28, 145, 95, 16.5, 100.5, 115, 1, 1, 0, 1, 1]])
        X = scaler.transform(patient)
        pred1 = model.predict(X)
        pred2 = model.predict(X)
        assert pred1[0] == pred2[0], "Model is non-deterministic"
 
    def test_high_risk_vitals_predict_high(self, trained_artifacts):
        model, scaler = trained_artifacts
        # Very dangerous vitals — should be high risk
        patient = np.array([[38, 160, 105, 19.0, 102.5, 125, 1, 1, 1, 1, 1]])
        X = scaler.transform(patient)
        pred = model.predict(X)[0]
        assert INV_LABEL[pred] == "high risk", \
            f"Expected high risk, got {INV_LABEL[pred]}"
 
    def test_safe_vitals_predict_low(self, trained_artifacts):
        model, scaler = trained_artifacts
        # Clearly safe vitals — should be low risk
        patient = np.array([[22, 108, 68, 7.0, 98.2, 68, 0, 0, 0, 0, 0]])
        X = scaler.transform(patient)
        pred = model.predict(X)[0]
        assert INV_LABEL[pred] == "low risk", \
            f"Expected low risk, got {INV_LABEL[pred]}"
 
    def test_output_shape_for_batch(self, trained_artifacts):
        model, scaler = trained_artifacts
        batch = np.random.rand(10, len(FEATURE_COLUMNS))
        X = scaler.transform(batch)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        assert preds.shape == (10,)
        assert probs.shape == (10, 3)
 
 
# ── Edge cases ────────────────────────────────────────────────────────────────
 
class TestEdgeCases:
 
    def test_zero_symptoms_does_not_crash(self, trained_artifacts):
        model, scaler = trained_artifacts
        patient = np.array([[25, 120, 80, 8.0, 98.6, 80, 0, 0, 0, 0, 0]])
        X = scaler.transform(patient)
        pred = model.predict(X)
        assert pred[0] in {0, 1, 2}
 
    def test_all_symptoms_does_not_crash(self, trained_artifacts):
        model, scaler = trained_artifacts
        patient = np.array([[40, 160, 110, 20.0, 103.0, 130, 1, 1, 1, 1, 1]])
        X = scaler.transform(patient)
        pred = model.predict(X)
        assert pred[0] in {0, 1, 2}
 
    def test_young_patient_does_not_crash(self, trained_artifacts):
        model, scaler = trained_artifacts
        patient = np.array([[15, 100, 65, 6.5, 98.0, 65, 0, 0, 0, 0, 0]])
        X = scaler.transform(patient)
        pred = model.predict(X)
        assert pred[0] in {0, 1, 2}
 
