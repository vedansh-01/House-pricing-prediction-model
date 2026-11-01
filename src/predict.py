"""
predict.py — Lightweight inference module.

Loads model + pipeline once (module-level singleton) and exposes predict().
This is what the FastAPI model_loader wraps.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PKL, FEATURE_PIPELINE_PKL


@lru_cache(maxsize=1)
def _load_artifacts():
    """Load and cache model + pipeline (called once at import)."""
    model = joblib.load(BEST_MODEL_PKL)
    fe = joblib.load(FEATURE_PIPELINE_PKL)
    return model, fe


def predict(input_dict: dict) -> float:
    """
    Run end-to-end inference for a single house.

    Args:
        input_dict: raw house features as a flat dict (pre-engineering)

    Returns:
        Predicted SalePrice in USD (float)
    """
    model, fe = _load_artifacts()
    input_df = pd.DataFrame([input_dict])
    X = fe.transform(input_df)
    y_log = model.predict(X)[0]
    return float(np.expm1(y_log))


def predict_batch(inputs: list[dict]) -> list[float]:
    """Batch inference for multiple houses."""
    model, fe = _load_artifacts()
    df = pd.DataFrame(inputs)
    X = fe.transform(df)
    y_log = model.predict(X)
    return [float(np.expm1(v)) for v in y_log]
