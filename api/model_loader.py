"""
model_loader.py — Load model artifacts once at startup (singleton pattern).

Loaded at module import time so there's zero per-request overhead.
Exposes model, feature_engineer and a predict() convenience function.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow running from project root or from api/ directory
_BASE = Path(__file__).resolve().parent.parent
_MODEL_PATH = _BASE / "models" / "best_model.pkl"
_PIPELINE_PATH = _BASE / "data" / "processed" / "feature_pipeline.pkl"

_model = None
_feature_engineer = None
_model_loaded = False
_pipeline_loaded = False


def _load() -> None:
    """Load model and pipeline into module-level singletons."""
    global _model, _feature_engineer, _model_loaded, _pipeline_loaded

    if _MODEL_PATH.exists():
        _model = joblib.load(_MODEL_PATH)
        _model_loaded = True
    else:
        print(f"[model_loader] WARNING: model not found at {_MODEL_PATH}")

    if _PIPELINE_PATH.exists():
        _feature_engineer = joblib.load(_PIPELINE_PATH)
        _pipeline_loaded = True
    else:
        print(f"[model_loader] WARNING: pipeline not found at {_PIPELINE_PATH}")


# Load immediately on import
_load()


def get_model():
    return _model


def get_feature_engineer():
    return _feature_engineer


def is_ready() -> bool:
    return _model_loaded and _pipeline_loaded


def predict(input_dict: dict) -> float:
    """
    Run inference for a single house using the loaded model + pipeline.
    Returns predicted SalePrice in USD.
    """
    if not is_ready():
        raise RuntimeError("Model or pipeline not loaded. Run training first.")
    df = pd.DataFrame([input_dict])
    X = _feature_engineer.transform(df)
    y_log = _model.predict(X)[0]
    return float(np.expm1(y_log))


def predict_batch(inputs: list[dict]) -> list[float]:
    """Batch inference."""
    if not is_ready():
        raise RuntimeError("Model or pipeline not loaded.")
    df = pd.DataFrame(inputs)
    X = _feature_engineer.transform(df)
    y_log = _model.predict(X)
    return [float(np.expm1(v)) for v in y_log]
