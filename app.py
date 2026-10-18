"""FastAPI service: GET /health, GET /model-info, POST /predict.

    uvicorn app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.config import DISCLAIMER
from src.inference import Predictor
from src.schemas import FarmRecord, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the saved models once at start-up."""
    try:
        app.state.predictor = Predictor()
    except FileNotFoundError:
        app.state.predictor = None
    yield


app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    description=DISCLAIMER,
    version="1.0.0",
    lifespan=lifespan,
)


def _predictor(request: Request) -> Predictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Models are not available. Run python run_all.py first.",
        )
    return predictor


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a short, safe 422 that names each bad field (no internals, no echo)."""
    problems = []
    for err in exc.errors():
        location = [str(p) for p in err["loc"] if p != "body"]
        problems.append({"field": ".".join(location) or "body", "message": err["msg"]})
    return JSONResponse(
        status_code=422,
        content={
            "error": "invalid_input",
            "message": "The farm record was rejected. Fix the listed fields and try again.",
            "problems": problems,
        },
    )


@app.get("/health")
def health(request: Request) -> dict:
    """Is the service up, and are the models loaded?"""
    predictor = request.app.state.predictor
    if predictor is None:
        return {"status": "models_missing"}
    return {"status": "ok", "verification_code": predictor.verification_code}


@app.get("/model-info")
def model_info(request: Request) -> dict:
    """Which models are served, what they were trained on, and the verification code."""
    meta = _predictor(request).metadata
    return {
        "service": meta["service"],
        "verification_code": meta["verification_code"],
        "synthetic_data": meta["synthetic_data"],
        "dataset_sha256": meta["dataset_sha256"],
        "rows_after_cleaning": meta["rows_after_cleaning"],
        "seed": meta["seed"],
        "input_features": meta["input_features"],
        "regression_model": meta["regression_model"],
        "classification_model": meta["classification_model"],
        "clusters": meta["clusters"],
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(record: FarmRecord, request: Request) -> dict:
    """Yield estimate, risk prediction, probability and cluster for one farm."""
    return _predictor(request).predict(record.model_dump())
