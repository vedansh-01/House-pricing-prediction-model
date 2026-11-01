"""
main.py — FastAPI application: prediction, batch, health, and SHAP explanation endpoints.

Run locally:
    uvicorn api.main:app --reload --port 8000

Docker:
    CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import model_loader
from api.schema import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    HouseFeatures,
    PredictionResponse,
)

# ── Lifespan (startup / shutdown) ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: model is already loaded at import, just log status
    status_str = "ready" if model_loader.is_ready() else "NOT READY — run training first"
    print(f"[startup] Model loader status: {status_str}")
    yield
    # Shutdown: nothing to teardown


# ── App ─────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="House Price Prediction API",
    description=(
        "End-to-end ML API for predicting Ames Housing sale prices. "
        "Submit raw house features → get predicted price in USD. "
        "Built with XGBoost/LightGBM, SHAP explainability, and MLflow tracking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ───────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.time() - t0) * 1000, 2))
    return response


# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Liveness check — returns model load status and version."""
    return HealthResponse(
        status="ok" if model_loader.is_ready() else "degraded",
        model_loaded=model_loader._model_loaded,
        pipeline_loaded=model_loader._pipeline_loaded,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict house sale price",
)
async def predict_price(features: HouseFeatures):
    """
    Submit raw house features and receive a predicted sale price in USD.

    - Input: all Ames Housing property features (most optional with sensible defaults)
    - Output: predicted_price (USD), currency, model_version
    - Errors: 422 for validation failures, 503 if model not loaded
    """
    if not model_loader.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please run training first.",
        )
    try:
        price = model_loader.predict(features.to_dict())
        return PredictionResponse(predicted_price=round(price, 2))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(exc)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Batch predict sale prices",
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict sale prices for multiple houses in a single request.
    Returns a list of predicted prices in the same order as input.
    """
    if not model_loader.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )
    try:
        inputs = [h.to_dict() for h in request.houses]
        prices = model_loader.predict_batch(inputs)
        return BatchPredictionResponse(
            predictions=[round(p, 2) for p in prices],
            count=len(prices),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(exc)}",
        )


@app.get("/", tags=["Root"])
async def root():
    """API root — returns welcome message and documentation link."""
    return {
        "message": "House Price Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }
