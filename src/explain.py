"""
explain.py — SHAP-based model explainability.

Usage:
    python -m src.explain

Produces:
    - data/processed/shap_summary.png  (global feature importance)
    - MLflow artifact: shap_summary.png
    - explain_single_prediction() for local force-plot explanations
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap

from src.config import (
    BEST_MODEL_PKL,
    DATA_PROCESSED,
    FEATURE_PIPELINE_PKL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    TARGET_COL,
)
from src.data_prep import load_data, remove_outliers

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_model_and_pipeline():
    """Load serialized model and feature pipeline."""
    model = joblib.load(BEST_MODEL_PKL)
    fe = joblib.load(FEATURE_PIPELINE_PKL)
    return model, fe


def generate_shap_summary(
    n_background: int = 500,
    save_path: str | None = None,
) -> shap.Explanation:
    """
    Generate SHAP summary plot for the best model.
    Uses TreeExplainer (fast, exact for XGBoost/LightGBM/RF).
    Falls back to KernelExplainer for linear models.

    Args:
        n_background: number of training samples to use as background
        save_path: optional PNG file path override

    Returns:
        SHAP Explanation object
    """
    model, fe = load_model_and_pipeline()
    train_df, _ = load_data()
    train_df = remove_outliers(train_df)

    y_log = np.log1p(train_df[TARGET_COL])
    X = fe.transform(train_df)

    # Sample background for speed
    bg_idx = np.random.RandomState(42).choice(len(X), min(n_background, len(X)), replace=False)
    X_bg = X.iloc[bg_idx]

    # Select explainer type based on model class
    model_type = type(model).__name__
    print(f"[explain] Using model: {model_type}")

    if hasattr(model, "predict") and model_type in (
        "XGBRegressor", "LGBMRegressor", "RandomForestRegressor",
        "GradientBoostingRegressor", "StackingRegressor"
    ):
        try:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        except Exception:
            try:
                explainer = shap.TreeExplainer(model, X_bg)
            except Exception:
                explainer = shap.KernelExplainer(model.predict, X_bg)
    else:
        explainer = shap.LinearExplainer(model, X_bg)

    print(f"[explain] Computing SHAP values on {len(X_bg)} samples ...")
    shap_values = explainer(X_bg)

    # ── Summary plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_bg,
        show=False,
        plot_size=None,
        max_display=20,
        color_bar_label="Feature value",
    )
    plt.title("SHAP Feature Importance — Global", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    out_path = save_path or str(DATA_PROCESSED / "shap_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[explain] SHAP summary saved → {out_path}")

    # Log to MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="SHAP_Explainability"):
        mlflow.log_artifact(out_path, artifact_path="shap")
        mlflow.log_metric("n_background_samples", len(X_bg))

    return shap_values


def explain_single_prediction(
    input_dict: dict,
    model=None,
    fe=None,
    plot: bool = True,
    save_path: str | None = None,
) -> dict:
    """
    Generate SHAP local explanation for a single prediction.

    Args:
        input_dict: raw house features as a dictionary (pre-engineering)
        model: optional pre-loaded model (will load from disk if None)
        fe: optional pre-loaded FeatureEngineer (will load from disk if None)
        plot: if True, show/save force plot
        save_path: optional path to save force plot PNG

    Returns:
        dict with 'predicted_price', 'shap_values', 'feature_names'
    """
    if model is None or fe is None:
        model, fe = load_model_and_pipeline()

    # Build single-row DataFrame
    input_df = pd.DataFrame([input_dict])
    X = fe.transform(input_df)

    # Predict
    y_log_pred = model.predict(X)[0]
    predicted_price = float(np.expm1(y_log_pred))

    # SHAP
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.KernelExplainer(model.predict, X)

    shap_values = explainer(X)

    if plot:
        # Bar plot of local SHAP values
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap_values[0], show=False, max_display=15)
        plt.title(
            f"SHAP Explanation — Predicted Price: ${predicted_price:,.0f}",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[explain] Force plot saved → {save_path}")
        plt.close(fig)

    # Top features sorted by |SHAP|
    sv = shap_values[0].values
    fn = shap_values[0].feature_names
    ranked = sorted(zip(fn, sv), key=lambda x: abs(x[1]), reverse=True)

    return {
        "predicted_price": predicted_price,
        "shap_feature_impacts": [
            {"feature": f, "shap_value": round(float(v), 4)} for f, v in ranked[:15]
        ],
    }


if __name__ == "__main__":
    print("[explain] Generating SHAP summary plot ...")
    generate_shap_summary()
    print("[explain] Done.")
