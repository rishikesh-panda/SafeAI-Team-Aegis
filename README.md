# SafeAI — Team Aegis Prototype

**SafeAI: Intelligent Public Safety Risk Prediction and Resource Allocation System using Machine Learning**  
Tagline: *Predict earlier. Plan smarter. Respond better.*

## Features
- Responsive dashboard and risk-assessment form
- Flask REST API connected to a trained scikit-learn Random Forest pipeline
- Risk classification: Low, Medium, High, Critical
- Estimated resource planning outputs
- SQLite-backed assessment history
- Model evaluation on a stratified holdout split
- Automatically generated synthetic demo dataset, so the prototype runs without finding external data first

## Important limitation
The included training data is synthetic and generated only to demonstrate the software workflow. Its accuracy measures performance against synthetic labels, not real-world public-safety performance. Risk scores and resource quantities are illustrative decision-support outputs. Do not use this prototype for real emergency or operational decisions. Replace demo data with credible, properly licensed data and obtain domain-expert validation before making real-world claims.

## Run on macOS
1. Extract the ZIP.
2. Open Terminal and change directory into `SafeAI_Best_Prototype`.
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the application:
   ```bash
   python backend/app.py
   ```
6. Open **http://127.0.0.1:5000**. Keep Terminal open while using it. Stop with Control+C.

On first start, the app creates a 1,200-row synthetic CSV, trains the model, reports the holdout accuracy, and creates the SQLite database.

## Structure
- `frontend/index.html` — interactive web interface
- `backend/app.py` — Flask API, model training and SQLite persistence
- `dataset/` — generated synthetic demo data (created on first run)
- `docs/ARCHITECTURE.md` — architecture notes
- `requirements.txt` — Python dependencies

## API
- `GET /api/health`
- `GET /api/metrics`
- `POST /api/predict`

Before GitHub submission, review `.gitignore`, add required team member/institution details, and confirm that your faculty accepts a synthetic-data prototype.
