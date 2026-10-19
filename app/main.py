"""Small inference service. Model loading errors prevent startup."""

import json
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt

ROOT = Path(__file__).resolve().parent.parent


# print("破坏测试")
def validate_values(values, features):
    if len(values) != len(features):
        raise ValueError(f"Expected {len(features)} values, received {len(values)}")
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Every feature must be a finite, nonnegative number")


@asynccontextmanager
async def lifespan(app):
    app.state.model = joblib.load(ROOT / "models/model.joblib")
    app.state.metadata = json.loads((ROOT / "models/metadata.json").read_text())
    yield


app = FastAPI(title="Vineyard Lab", lifespan=lifespan)


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: list[StrictFloat | StrictInt]


@app.get("/")
def index():
    return FileResponse(ROOT / "app/index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": os.getenv("APP_ENV", "local"),
        "run_id": app.state.metadata["run_id"],
    }


@app.get("/model")
def metadata():
    return app.state.metadata


@app.post("/predict")
def predict(sample: Sample):
    features = app.state.metadata["feature_names"]
    try:
        validate_values(sample.values, features)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    frame = pd.DataFrame([sample.values], columns=features)
    # Very large finite values can overflow preprocessing; reject those too.
    if max(sample.values) > 1e9:
        raise HTTPException(422, "Feature is outside supported numerical range")
    probabilities = app.state.model.predict_proba(frame)[0]
    return {
        "cultivar": int(app.state.model.classes_[probabilities.argmax()]) + 1,
        "probabilities": [float(v) for v in probabilities],
        "run_id": app.state.metadata["run_id"],
        "note": "Educational classifier; probabilities are not calibrated confidence.",
    }
