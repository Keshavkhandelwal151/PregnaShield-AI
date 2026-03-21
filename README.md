# PregnaShield-AI
PregnaShield AI is an AI-powered telecare system that monitors pregnant women using health data and symptoms. It predicts risk levels, prioritizes high-risk cases, and alerts doctors for early intervention, improving maternal care in rural and underserved areas.
Copy🤰 Maternal Telecare Risk Triage System
Show Image
Show Image
Show Image
Show Image
Show Image
An AI-powered telecare system that collects remote health data from pregnant women, predicts pregnancy risk using machine learning, and alerts healthcare workers in real time.

📋 Problem Statement
Many pregnant women in rural/remote areas miss regular checkups. High-risk pregnancies go undetected, leading to preventable complications. This system automates risk triage so healthcare workers can prioritize critical patients instantly.

🏗️ System Architecture
Mobile App / Wearable Device
        ↓
Health Data Collection (BP, HR, Glucose, Symptoms)
        ↓
FastAPI Backend → PostgreSQL Database
        ↓
ML Risk Prediction Model
        ↓
Risk Score → Triage (High / Medium / Low)
        ↓
Streamlit Dashboard → Alert to Healthcare Worker
        ↓
Teleconsultation

🗂️ Project Structure
maternal-telecare-risk-triage/
├── README.md
├── requirements.txt
├── .gitignore
├── docker-compose.yml
├── ml/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   ├── evaluate.py
│   └── models/
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── models/schemas.py
│   ├── db/database.py
│   └── services/
├── dashboard/
│   └── app.py
├── data/
│   ├── maternal_health_risk.csv
│   └── sample_patients.csv
└── tests/
    ├── test_risk_engine.py
    ├── test_api.py
    └── test_model.py

⚙️ Setup & Installation
1. Clone the repository
bashgit clone https://github.com/yourusername/maternal-telecare-risk-triage.git
cd maternal-telecare-risk-triage
2. Create virtual environment
bashpython -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
3. Install dependencies
bashpip install -r requirements.txt
4. Set environment variables
bashcp .env.example .env
# Edit .env with your database URL and API keys
5. Train the ML model
bashpython ml/train_model.py
6. Run the backend API
bashuvicorn backend.main:app --reload
7. Run the dashboard
bashstreamlit run dashboard/app.py

🤖 ML Model

Dataset: UCI Maternal Health Risk Dataset + Synthetic data
Features: Age, Blood Pressure, Blood Sugar, Heart Rate, Body Temp, Symptoms
Models Tried: Logistic Regression, Random Forest, Gradient Boosting
Output: Risk Score (0–1) → Category: Low / Medium / High

Risk ScoreCategoryAction0.0 – 0.4🟢 LowRoutine monitoring0.4 – 0.7🟡 MediumSchedule teleconsultation0.7 – 1.0🔴 HighImmediate doctor alert

🚀 API Endpoints
MethodEndpointDescriptionPOST/predictSubmit vitals, get risk scoreGET/patientsList all patientsGET/patients/{id}Get patient detailsPOST/alertTrigger alert to healthcare worker
API docs available at: http://localhost:8000/docs

📊 Dataset Sources

UCI Maternal Health Risk
Kaggle Maternal Health


🧪 Running Tests
bashpytest tests/ -v

🌟 Expected Impact

Reduce maternal mortality in rural areas
Early detection of gestational diabetes, preeclampsia, anemia
Help health workers prioritize critical patients
Enable remote consultations via telemedicine


📄 License
MIT License — free for academic and research use.
