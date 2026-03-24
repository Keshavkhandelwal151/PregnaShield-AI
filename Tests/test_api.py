"""
tests/test_api.py
-----------------
Integration tests for FastAPI endpoints using TestClient.
Run with: pytest tests/test_api.py -v
"""
 
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))
 
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
 
from main import app
 
client = TestClient(app)
 
 
# ── Sample payloads ───────────────────────────────────────────────────────────
 
HIGH_RISK_PAYLOAD = {
    "patient_id"            : "TEST001",
    "age"                   : 35,
    "systolic_bp"           : 150,
    "diastolic_bp"          : 100,
    "blood_sugar"           : 17.5,
    "body_temp"             : 101.5,
    "heart_rate"            : 118,
    "headache"              : 1,
    "swelling"              : 1,
    "bleeding"              : 1,
    "abdominal_pain"        : 1,
    "reduced_fetal_movement": 1,
}
 
LOW_RISK_PAYLOAD = {
    "patient_id"            : "TEST002",
    "age"                   : 22,
    "systolic_bp"           : 110,
    "diastolic_bp"          : 70,
    "blood_sugar"           : 7.5,
    "body_temp"             : 98.6,
    "heart_rate"            : 72,
    "headache"              : 0,
    "swelling"              : 0,
    "bleeding"              : 0,
    "abdominal_pain"        : 0,
    "reduced_fetal_movement": 0,
}
 
 
# ── Health endpoints ──────────────────────────────────────────────────────────
 
class TestHealthEndpoints:
 
    def test_root_returns_ok(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
 
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
 
 
# ── Predict endpoint ──────────────────────────────────────────────────────────
 
class TestPredictEndpoint:
 
    @patch("routes.predict.get_risk_prediction")
    def test_predict_high_risk(self, mock_predict):
        mock_predict.return_value = {
            "risk_score"    : 0.91,
            "risk_category" : "high risk",
            "action"        : "Immediate doctor alert",
            "probabilities" : {"low risk": 0.02, "mid risk": 0.07, "high risk": 0.91},
        }
        response = client.post("/predict/", json=HIGH_RISK_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_category"] == "high risk"
        assert data["risk_score"]    == 0.91
        assert "action"              in data
        assert "probabilities"       in data
        assert "timestamp"           in data
 
    @patch("routes.predict.get_risk_prediction")
    def test_predict_low_risk(self, mock_predict):
        mock_predict.return_value = {
            "risk_score"    : 0.08,
            "risk_category" : "low risk",
            "action"        : "Routine monitoring",
            "probabilities" : {"low risk": 0.88, "mid risk": 0.10, "high risk": 0.02},
        }
        response = client.post("/predict/", json=LOW_RISK_PAYLOAD)
        assert response.status_code == 200
        assert response.json()["risk_category"] == "low risk"
 
    def test_predict_missing_required_field(self):
        bad_payload = {k: v for k, v in HIGH_RISK_PAYLOAD.items() if k != "age"}
        response = client.post("/predict/", json=bad_payload)
        assert response.status_code == 422   # Unprocessable Entity
 
    def test_predict_invalid_bp_value(self):
        payload = HIGH_RISK_PAYLOAD.copy()
        payload["systolic_bp"] = 999          # out of range
        response = client.post("/predict/", json=payload)
        assert response.status_code == 422
 
    def test_predict_response_has_patient_id(self):
        with patch("routes.predict.get_risk_prediction") as mock_predict:
            mock_predict.return_value = {
                "risk_score"    : 0.5,
                "risk_category" : "mid risk",
                "action"        : "Schedule teleconsultation",
                "probabilities" : {"low risk": 0.2, "mid risk": 0.6, "high risk": 0.2},
            }
            response = client.post("/predict/", json=LOW_RISK_PAYLOAD)
            assert response.json()["patient_id"] == LOW_RISK_PAYLOAD["patient_id"]
 
 
# ── Patients endpoint ─────────────────────────────────────────────────────────
 
class TestPatientsEndpoint:
 
    def test_list_patients_returns_list(self):
        response = client.get("/patients/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
 
    def test_create_patient_success(self):
        payload = {
            "name"          : "Test Patient",
            "age"           : 28,
            "phone"         : "+911234567890",
            "location"      : "Delhi",
            "weeks_pregnant": 24,
        }
        response = client.post("/patients/", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Patient"
        assert "id" in data
 
    def test_create_patient_missing_name(self):
        payload = {"age": 28}
        response = client.post("/patients/", json=payload)
        assert response.status_code == 422
 
    def test_get_patient_not_found(self):
        response = client.get("/patients/DOESNOTEXIST")
        assert response.status_code == 404
 
    def test_get_patient_after_create(self):
        create_payload = {"name": "Lookup Test", "age": 30}
        create_resp = client.post("/patients/", json=create_payload)
        patient_id  = create_resp.json()["id"]
 
        get_resp = client.get(f"/patients/{patient_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["name"] == "Lookup Test"
 
 
# ── Alerts endpoint ───────────────────────────────────────────────────────────
 
class TestAlertsEndpoint:
 
    def test_trigger_high_risk_alert(self):
        payload = {
            "patient_id"    : "P001",
            "risk_category" : "high risk",
            "risk_score"    : 0.91,
            "message"       : "Immediate attention needed.",
        }
        response = client.post("/alerts/", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["status"]     == "sent"
        assert data["patient_id"] == "P001"
        assert "alert_id" in data
 
    def test_trigger_mid_risk_alert(self):
        payload = {
            "patient_id"    : "P003",
            "risk_category" : "mid risk",
            "risk_score"    : 0.55,
        }
        response = client.post("/alerts/", json=payload)
        assert response.status_code == 201
 
    def test_low_risk_alert_rejected(self):
        payload = {
            "patient_id"    : "P002",
            "risk_category" : "low risk",
            "risk_score"    : 0.12,
        }
        response = client.post("/alerts/", json=payload)
        assert response.status_code == 400   # alerts not sent for low risk
 
