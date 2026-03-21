"""
schemas.py
----------
Pydantic models for request validation and response serialization.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime


# ── Prediction ────────────────────────────────────────────────────────────────

class VitalsInput(BaseModel):
    """Input payload from mobile app or wearable device."""
    patient_id   : str   = Field(..., example="P001")
    age          : int   = Field(..., ge=10, le=60, example=28)
    systolic_bp  : int   = Field(..., ge=70, le=200, example=140)
    diastolic_bp : int   = Field(..., ge=40, le=130, example=90)
    blood_sugar  : float = Field(..., ge=5.0, le=25.0, example=15.5)
    body_temp    : float = Field(..., ge=95.0, le=106.0, example=100.5)
    heart_rate   : int   = Field(..., ge=40, le=160, example=110)

    # Patient-reported symptoms (1 = yes, 0 = no)
    headache              : int = Field(0, ge=0, le=1)
    swelling              : int = Field(0, ge=0, le=1)
    bleeding              : int = Field(0, ge=0, le=1)
    abdominal_pain        : int = Field(0, ge=0, le=1)
    reduced_fetal_movement: int = Field(0, ge=0, le=1)


class RiskPredictionResponse(BaseModel):
    patient_id    : str
    risk_score    : float
    risk_category : str
    action        : str
    probabilities : Dict[str, float]
    timestamp     : datetime


# ── Patient ───────────────────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    name         : str
    age          : int
    phone        : Optional[str] = None
    location     : Optional[str] = None
    weeks_pregnant: Optional[int] = None


class PatientResponse(PatientCreate):
    id            : str
    risk_category : Optional[str] = None
    risk_score    : Optional[float] = None
    last_updated  : Optional[datetime] = None


# ── Alert ─────────────────────────────────────────────────────────────────────

class AlertRequest(BaseModel):
    patient_id    : str
    risk_category : str
    risk_score    : float
    message       : Optional[str] = None


class AlertResponse(BaseModel):
    alert_id   : str
    patient_id : str
    status     : str
    sent_at    : datetime
