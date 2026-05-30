"""FastAPI inference server loading the latest registered MLflow model."""
import mlflow.pyfunc, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="MLOps Template Inference")
MODEL = mlflow.pyfunc.load_model("models:/rf_classifier/latest")

class Batch(BaseModel):
    rows: List[dict]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(batch: Batch):
    df = pd.DataFrame(batch.rows)
    preds = MODEL.predict(df)
    return {"predictions": [int(p) for p in preds]}
