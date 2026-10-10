from pathlib import Path
from datetime import datetime
import sqlite3, random
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT=Path(__file__).resolve().parents[1]
DB=ROOT/"backend"/"safeai.db"
MODEL=ROOT/"backend"/"model.joblib"
DATA=ROOT/"dataset"/"demo_training_data.csv"
NUM=["crowd_size","venue_capacity","temperature_c","holiday","previous_incidents"]
CAT=["weather","time_slot","day_of_week","event_type"]
FEATURES=NUM+CAT
app=Flask(__name__,static_folder=str(ROOT/"frontend"),static_url_path="")

def ensure_data():
    if DATA.exists(): return
    random.seed(42); rows=[]
    weathers=["Clear","Cloudy","Rain","Heavy Rain","Extreme Heat"]
    events=["Normal Operations","Sports Event","Concert","Festival","Public Gathering"]
    times=["Morning","Afternoon","Evening","Night"]
    days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    for _ in range(1200):
        cap=random.randint(1000,20000); crowd=random.randint(100,int(cap*1.25))
        inc=random.randint(0,12); temp=random.randint(15,44); weather=random.choice(weathers)
       
        event = random.choice(events)
        holiday = random.randint(0, 1)
        time = random.randint(0, 3)
        day = random.choice(days)

        
        score = 5 + (crowd / cap) * 35
        score += inc * 3
        score += holiday * 3

        score += {
            "Clear": 0,
            "Cloudy": 2,
            "Rain": 6,
            "Heavy Rain": 12,
            "Extreme Heat": 10
        }[weather]

        score += {
            "Normal Operations": 0,
            "Sports Event": 7,
            "Concert": 10,
            "Festival": 9,
            "Public Gathering": 6
        }[event]

        score += [0, 5, 11, 3][time]
        score += 4 if day in ["Saturday", "Sunday"] else 0
        score += 6 if temp >= 39 else 0

        score = max(0, min(100, score + random.gauss(0, 3)))

        risk = (
            "Low" if score < 40
            else "Medium" if score < 60
            else "High" if score < 80
            else "Critical"
        )

        rows.append([crowd, cap, temp, holiday, inc, weather, time, day, event, risk])

    pd.DataFrame(rows, columns=FEATURES + ["risk_level"]).to_csv(DATA, index=False)


def train():
    ensure_data()
    df = pd.read_csv(DATA)
    X = df[FEATURES]
    y = df["risk_level"]
    print("Dataset columns:", df.columns.tolist())
    print("Numeric columns:", NUM)
    print("Categorical columns:", CAT)
    print("First row:", X.iloc[0].to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    
    prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())
                ]),
                NUM
            ),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"))
                ]),
                CAT
            )
        ]
    )


    model = Pipeline([
        ("prep", prep),
        ("rf", RandomForestClassifier(
            n_estimators=180,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    acc = float(accuracy_score(y_test, model.predict(X_test)))

    joblib.dump(
        {"model": model, "accuracy": acc, "rows": len(df)},
        MODEL
    )
    return model, acc, len(df)


def get_model():
    if not MODEL.exists(): return train()
    p=joblib.load(MODEL); return p["model"],p["accuracy"],p["rows"]

def init_db():
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS assessments(id INTEGER PRIMARY KEY AUTOINCREMENT,created TEXT,venue TEXT,crowd INTEGER,weather TEXT,event TEXT,risk TEXT,score INTEGER)""")
init_db()

@app.get("/")
def home(): return send_from_directory(ROOT/"frontend","index.html")
@app.get("/api/health")
def health(): return jsonify(status="ok",service="SafeAI")
@app.get("/api/metrics")
def metrics():
    m,acc,n=get_model()
    with sqlite3.connect(DB) as c:
        total=c.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
        high=c.execute("SELECT COUNT(*) FROM assessments WHERE risk IN ('High','Critical')").fetchone()[0]
        rows=c.execute("SELECT id,created,venue,crowd,weather,event,risk,score FROM assessments ORDER BY id DESC LIMIT 30").fetchall()
    keys=["id","created","venue","crowd","weather","event","risk","score"]
    return jsonify(accuracy=round(acc*100,1),training_rows=n,total=total,high_risk=high,history=[dict(zip(keys,r)) for r in rows])
@app.post("/api/predict")
def predict():
    x=request.get_json(silent=True) or {}
    try:
        venue=str(x.get("venue","Unnamed location")).strip()[:100] or "Unnamed location"
        r={"crowd_size":int(x["crowd_size"]),"venue_capacity":int(x["venue_capacity"]),"weather":str(x["weather"]),
           "temperature_c":float(x["temperature_c"]),"time_slot":str(x["time_slot"]),"day_of_week":str(x["day_of_week"]),
           "holiday":1 if str(x["holiday"]).lower() in ("yes","1","true") else 0,"event_type":str(x["event_type"]),
           "previous_incidents":int(x["previous_incidents"])}
        if r["crowd_size"]<0 or r["venue_capacity"]<=0 or r["previous_incidents"]<0: raise ValueError("Crowd/incidents must be non-negative and capacity must be positive.")
        m, acc, n = get_model()
        frame = pd.DataFrame([r], columns=FEATURES)        
        confidence=round(float(max(m.predict_proba(frame)[0]))*100,1)
        class_scores = {"Low": 20, "Medium": 50, "High": 70, "Critical": 90}
        probabilities = m.predict_proba(frame)[0]
        score = round(sum(float(p) * class_scores.get(str(label), 50) for label, p in zip(m.classes_, probabilities)))
        risk = "Low" if score < 40 else "Medium" if score < 60 else "High" if score < 80 else "Critical"
        
        mult={"Low":.65,"Medium":.85,"High":1.15,"Critical":1.4}[risk]; crowd=r["crowd_size"]
        resources={"security_personnel":max(2,round(crowd/500*mult)),"police_units":max(1,round(crowd/1800*mult)),
                   "medical_teams":max(1,round(crowd/3500*mult)),"ambulances":max(1,round(crowd/5000*mult))}
        now=datetime.now().astimezone().isoformat(timespec="seconds")
        with sqlite3.connect(DB) as c:
            cur=c.execute("INSERT INTO assessments(created,venue,crowd,weather,event,risk,score) VALUES(?,?,?,?,?,?,?)",
                (now,venue,crowd,r["weather"],r["event_type"],risk,score)); aid=cur.lastrowid
        return jsonify(id=aid,created=now,venue=venue,risk=risk,score=score,confidence=confidence,resources=resources,
                       note="Synthetic-data academic demo only; not for real safety decisions.")
    except (ValueError,KeyError,TypeError) as e: return jsonify(error=str(e)),400
if __name__=="__main__":
    m,a,n=get_model(); print(f"SafeAI ready: {n} synthetic rows; holdout accuracy {a*100:.1f}%")
    app.run(host="127.0.0.1",port=5000,debug=False)