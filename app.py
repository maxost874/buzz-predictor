from fastapi import FastAPI
from pydantic import BaseModel
import joblib, datetime as dt

# sentiment analyzer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# load model
MODEL_PATH = "pipeline_v1.joblib"
pipe = joblib.load(MODEL_PATH)
sia = SentimentIntensityAnalyzer()

app = FastAPI(title="Buzz Predictor")

class PredictRequest(BaseModel):
    text: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text

    pred  = int(pipe.predict([text])[0])
    proba = float(pipe.predict_proba([text])[0, 1])

    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "text": text,
        "prediction": pred,
        "probability": round(proba, 4),
        "sentiment": sentiment,
        "sentiment_scores": scores,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "model_version": "tfidf_logreg_v1"
    }
