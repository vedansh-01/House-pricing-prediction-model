"""
evaluate.py — Metrics and residual analysis utilities.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/script use
import matplotlib.pyplot as plt
import seaborn as sns


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Log Error (RMSLE).
    Expects raw SalePrice values (not already log-transformed).
    """
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(np.clip(y_pred, 0, None))) ** 2)))


def rmse_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """RMSE when both inputs are already in log1p scale (matches Kaggle metric)."""
    return float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))


def mae_dollars(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Mean Absolute Error in original dollar scale (inverse log transform)."""
    return float(np.mean(np.abs(np.expm1(y_true_log) - np.expm1(y_pred_log))))


def compute_metrics(
    y_true_log: np.ndarray, y_pred_log: np.ndarray, prefix: str = ""
) -> dict:
    """Return dict of all evaluation metrics."""
    prefix = f"{prefix}_" if prefix else ""
    return {
        f"{prefix}rmse_log": rmse_log(y_true_log, y_pred_log),
        f"{prefix}mae_dollars": mae_dollars(y_true_log, y_pred_log),
        f"{prefix}r2": float(1 - np.var(y_true_log - y_pred_log) / np.var(y_true_log)),
    }


def residual_analysis(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Generate residual analysis plots:
      - Residuals vs predicted (homoscedasticity check)
      - Histogram of residuals (normality check)
    """
    residuals = y_true_log - y_pred_log
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Residual Analysis", fontsize=14, fontweight="bold")

    # Residuals vs predicted
    axes[0].scatter(y_pred_log, residuals, alpha=0.4, edgecolors="none", s=20, color="#4C72B0")
    axes[0].axhline(0, color="red", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Predicted log(SalePrice)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")

    # Histogram of residuals
    axes[1].hist(residuals, bins=40, edgecolor="white", color="#4C72B0", alpha=0.85)
    axes[1].axvline(0, color="red", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[residual_analysis] Saved to {save_path}")
    return fig
