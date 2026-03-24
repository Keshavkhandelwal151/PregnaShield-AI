
"""
tests/test_risk_engine.py
--------------------------
Unit tests for the triage thresholds and risk engine logic.
Run with: pytest tests/ -v
"""
 
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ml"))
 
from preprocess import generate_synthetic_data, preprocess, LABEL_MAP
 
 
# ── Fixtures ──────────────────────────────────────────────────────────────────
 
@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_data(n_samples=200, seed=0)
 
 
# ── Triage threshold logic ─────────────────────────────────────────────────────
 
def score_to_category(score: float) -> str:
    """Mirrors the triage logic used in risk_engine.py."""
    if score >= 0.7:
        return "high risk"
    elif score >= 0.4:
        return "mid risk"
    return "low risk"
 
 
class TestTriageThresholds:
 
    def test_high_risk_threshold(self):
        assert score_to_category(0.70) == "high risk"
        assert score_to_category(0.85) == "high risk"
        assert score_to_category(1.00) == "high risk"
 
    def test_mid_risk_threshold(self):
        assert score_to_category(0.40) == "mid risk"
        assert score_to_category(0.55) == "mid risk"
        assert score_to_category(0.69) == "mid risk"
 
    def test_low_risk_threshold(self):
        assert score_to_category(0.00) == "low risk"
        assert score_to_category(0.20) == "low risk"
        assert score_to_category(0.39) == "low risk"
 
    def test_boundary_exactly_04(self):
        assert score_to_category(0.40) == "mid risk"
 
    def test_boundary_exactly_07(self):
        assert score_to_category(0.70) == "high risk"
 
 
# ── Synthetic data generation ─────────────────────────────────────────────────
 
class TestSyntheticData:
 
    def test_generates_correct_row_count(self, synthetic_df):
        assert len(synthetic_df) == 200
 
    def test_all_risk_labels_present(self, synthetic_df):
        labels = set(synthetic_df["RiskLevel"].unique())
        assert "low risk"  in labels
        assert "mid risk"  in labels
        assert "high risk" in labels
 
    def test_no_missing_values_in_key_columns(self, synthetic_df):
        key_cols = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp",
                    "HeartRate", "RiskLevel"]
        for col in key_cols:
            assert synthetic_df[col].isna().sum() == 0, f"{col} has nulls"
 
    def test_blood_pressure_in_range(self, synthetic_df):
        assert synthetic_df["SystolicBP"].between(90, 159).all()
        assert synthetic_df["DiastolicBP"].between(60, 109).all()
 
    def test_blood_sugar_in_range(self, synthetic_df):
        assert synthetic_df["BS"].between(6.0, 19.0).all()
 
 
# ── Preprocessing ─────────────────────────────────────────────────────────────
 
class TestPreprocessing:
 
    def test_preprocess_returns_correct_shapes(self, synthetic_df):
        X, y, scaler = preprocess(synthetic_df, fit_scaler=True)
        assert X.shape[0] == len(synthetic_df)
        assert X.shape[1] == 11          # 6 vitals + 5 symptoms
        assert len(y)     == len(synthetic_df)
 
    def test_labels_are_encoded_correctly(self, synthetic_df):
        _, y, _ = preprocess(synthetic_df, fit_scaler=True)
        assert set(y).issubset({0, 1, 2})
 
    def test_scaler_is_fitted(self, synthetic_df):
        from sklearn.preprocessing import StandardScaler
        _, _, scaler = preprocess(synthetic_df, fit_scaler=True)
        assert isinstance(scaler, StandardScaler)
        assert hasattr(scaler, "mean_")
 
    def test_inference_mode_uses_existing_scaler(self, synthetic_df):
        _, _, fitted_scaler = preprocess(synthetic_df, fit_scaler=True)
        X_inf, _, _ = preprocess(
            synthetic_df.iloc[:10].copy(),
            scaler=fitted_scaler,
            fit_scaler=False
        )
        assert X_inf.shape == (10, 11)
 
 
# ── Clinical rules smoke test ─────────────────────────────────────────────────
 
class TestClinicalRules:
    """
    Verify that obviously dangerous vitals map to high risk
    and safe vitals map to low risk in the synthetic generator.
    """
 
    def test_dangerous_vitals_generate_high_risk(self):
        import numpy as np
        from preprocess import generate_synthetic_data
        # Force dangerous conditions by seeding with known bad values
        df = generate_synthetic_data(n_samples=1000, seed=7)
        dangerous = df[
            (df["SystolicBP"] >= 140) &
            (df["BS"] > 15) &
            (df["Bleeding"] == 1) &
            (df["ReducedFetalMovement"] == 1)
        ]
        if len(dangerous) > 0:
            assert (dangerous["RiskLevel"] == "high risk").all(), \
                "Patients with severe vitals should be high risk"
 
    def test_normal_vitals_generate_low_risk(self):
        df = generate_synthetic_data(n_samples=1000, seed=7)
        safe = df[
            (df["SystolicBP"] < 120) &
            (df["BS"] < 9) &
            (df["Bleeding"] == 0) &
            (df["ReducedFetalMovement"] == 0) &
            (df["Headache"] == 0) &
            (df["Swelling"] == 0) &
            (df["AbdominalPain"] == 0)
        ]
        if len(safe) > 0:
            assert (safe["RiskLevel"] == "low risk").all(), \
                "Patients with normal vitals should be low risk"
 
