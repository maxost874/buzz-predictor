from fastapi import FastAPI
from pydantic import BaseModel
import joblib, datetime as dt

MODEL_PATH = "pipeline_v1.joblib"
app = FastAPI(title="Buzz Predictor")

pipe = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    text: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    pred  = int(pipe.predict([req.text])[0])
    proba = float(pipe.predict_proba([req.text])[0, 1])
    return {
        "text": req.text,
        "prediction": pred,
        "probability": proba,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z"
    }
