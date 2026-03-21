"""
main.py
-------
FastAPI application entry point.
Run with: uvicorn backend.main:app --reload
Docs at:  http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.predict import router as predict_router
from routes.patients import router as patients_router
from routes.alerts import router as alerts_router

app = FastAPI(
    title="Maternal Telecare Risk Triage API",
    description="AI-powered pregnancy risk prediction and triage system",
    version="1.0.0",
)

# ── CORS (allow Streamlit dashboard and mobile app) ───────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────────────────
app.include_router(predict_router,  prefix="/predict",  tags=["Risk Prediction"])
app.include_router(patients_router, prefix="/patients", tags=["Patients"])
app.include_router(alerts_router,   prefix="/alerts",   tags=["Alerts"])


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Maternal Telecare API is running"}


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}
