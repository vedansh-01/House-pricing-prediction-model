"""
train.py — Model training, cross-validation, and experiment tracking.

Usage:
    python -m src.train

Produces:
    - MLflow runs for every model × fold combination
    - models/best_model.pkl  (best single model by CV RMSE)
    - data/processed/feature_pipeline.pkl  (fitted on full train data)
    - Model comparison table printed to stdout
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from xgboost import XGBRegressor

from src.config import (
    BEST_MODEL_PKL,
    DATA_PROCESSED,
    FEATURE_PIPELINE_PKL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_FOLDS,
    RANDOM_STATE,
    TARGET_COL,
)
from src.data_prep import get_cv_folds, load_data, remove_outliers
from src.evaluate import compute_metrics
from src.feature_eng import FeatureEngineer

warnings.filterwarnings("ignore")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ── Model registry ─────────────────────────────────────────────────────────────

def get_models() -> dict:
    """Return all candidate models keyed by name."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.001, max_iter=5000, random_state=RANDOM_STATE),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_features="sqrt", min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", n_jobs=-1,
            random_state=RANDOM_STATE, verbosity=0
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=RANDOM_STATE, verbose=-1
        ),
    }


LINEAR_MODELS = {"LinearRegression", "Ridge", "Lasso", "ElasticNet"}


# ── Cross-validation ───────────────────────────────────────────────────────────

def cross_validate_model(
    model,
    model_name: str,
    df_train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """
    Run k-fold CV for a single model.
    Fits FeatureEngineer fresh inside each fold (no leakage).
    Returns dict with per-fold metrics and aggregate stats.
    """
    use_scaler = model_name in LINEAR_MODELS
    fold_rmse = []
    fold_mae = []

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        fold_train = df_train.iloc[tr_idx].copy()
        fold_val = df_train.iloc[val_idx].copy()

        y_tr = np.log1p(fold_train[TARGET_COL])
        y_val = np.log1p(fold_val[TARGET_COL])

        # Fit feature engineer on train fold only
        fe = FeatureEngineer(use_scaler=use_scaler)
        X_tr = fe.fit_transform(fold_train, y=y_tr)
        X_val = fe.transform(fold_val)

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        metrics = compute_metrics(y_val.values, y_pred, prefix=f"fold{fold_idx}")
        fold_rmse.append(metrics[f"fold{fold_idx}_rmse_log"])
        fold_mae.append(metrics[f"fold{fold_idx}_mae_dollars"])

    return {
        "model": model_name,
        "mean_rmse_log": float(np.mean(fold_rmse)),
        "std_rmse_log": float(np.std(fold_rmse)),
        "mean_mae_dollars": float(np.mean(fold_mae)),
        "fold_rmse": fold_rmse,
    }


# ── MLflow logging ─────────────────────────────────────────────────────────────

def log_cv_run(
    model_name: str, model, cv_results: dict, train_time: float
) -> None:
    """Log CV metrics and model params to MLflow."""
    with mlflow.start_run(run_name=f"{model_name}_CV"):
        mlflow.log_param("model", model_name)
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_params(model.get_params() if hasattr(model, "get_params") else {})
        mlflow.log_metric("mean_rmse_log", cv_results["mean_rmse_log"])
        mlflow.log_metric("std_rmse_log", cv_results["std_rmse_log"])
        mlflow.log_metric("mean_mae_dollars", cv_results["mean_mae_dollars"])
        mlflow.log_metric("train_time_seconds", train_time)
        for i, r in enumerate(cv_results["fold_rmse"]):
            mlflow.log_metric(f"fold_{i}_rmse_log", r)


# ── Main training loop ─────────────────────────────────────────────────────────

def train_all_models(include_stacking: bool = True) -> pd.DataFrame:
    """
    Train all models with 5-fold CV, log to MLflow, serialize best model.
    Returns comparison DataFrame sorted by mean CV RMSE.
    """
    print("\n" + "=" * 60)
    print("  HOUSE PRICE PREDICTION — TRAINING")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────────────────────
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)

    train_df, _ = load_data()
    train_df = remove_outliers(train_df)
    print(f"\n[train] After outlier removal: {train_df.shape}")

    folds = get_cv_folds(train_df, k=N_FOLDS)
    models = get_models()

    # ── MLflow experiment ───────────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    results = []
    for name, model in models.items():
        print(f"\n→ Training {name} ...")
        t0 = time.time()
        cv_res = cross_validate_model(model, name, train_df, folds)
        elapsed = time.time() - t0
        cv_res["train_time_s"] = round(elapsed, 2)
        results.append(cv_res)
        log_cv_run(name, model, cv_res, elapsed)
        print(
            f"  {name}: RMSE={cv_res['mean_rmse_log']:.4f} "
            f"(±{cv_res['std_rmse_log']:.4f})  "
            f"MAE=${cv_res['mean_mae_dollars']:,.0f}  "
            f"time={elapsed:.1f}s"
        )

    # ── Optional stacking ───────────────────────────────────────────────────────
    if include_stacking:
        print("\n→ Training StackingRegressor (LightGBM + XGBoost + Ridge meta) ...")
        estimators = [
            ("lgb", LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=RANDOM_STATE)),
            ("xgb", XGBRegressor(n_estimators=300, learning_rate=0.05, verbosity=0, random_state=RANDOM_STATE)),
        ]
        stacked = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=10.0),
            cv=3,
            n_jobs=-1,
        )
        t0 = time.time()
        try:
            cv_res_stack = cross_validate_model(stacked, "Stacking", train_df, folds)
            elapsed = time.time() - t0
            cv_res_stack["train_time_s"] = round(elapsed, 2)
            results.append(cv_res_stack)
            log_cv_run("Stacking", stacked, cv_res_stack, elapsed)
            print(
                f"  Stacking: RMSE={cv_res_stack['mean_rmse_log']:.4f} "
                f"(±{cv_res_stack['std_rmse_log']:.4f})  time={elapsed:.1f}s"
            )
        except Exception as e:
            print(f"  Stacking failed: {e}")

    # ── Model comparison table ──────────────────────────────────────────────────
    df_results = (
        pd.DataFrame(results)
        .drop(columns=["fold_rmse"], errors="ignore")
        .sort_values("mean_rmse_log")
        .reset_index(drop=True)
    )
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON TABLE")
    print("=" * 60)
    print(df_results.to_string(index=False))

    # ── Refit best model on full training data ──────────────────────────────────
    best_name = df_results.iloc[0]["model"]
    best_model = {**models}
    if include_stacking:
        stacked_models = {"Stacking": stacked}
        best_model.update(stacked_models)

    all_models_dict = get_models()
    if "Stacking" in best_name:
        best_model_obj = stacked
    else:
        best_model_obj = all_models_dict.get(best_name, list(all_models_dict.values())[0])

    print(f"\n[train] Best model: {best_name} — refitting on full training data ...")
    use_scaler = best_name in LINEAR_MODELS

    # Fit feature engineer on all training data
    y_full = np.log1p(train_df[TARGET_COL])
    fe_full = FeatureEngineer(use_scaler=use_scaler)
    X_full = fe_full.fit_transform(train_df, y=y_full)

    best_model_obj.fit(X_full, y_full)

    # Serialize
    joblib.dump(fe_full, FEATURE_PIPELINE_PKL)
    joblib.dump(best_model_obj, BEST_MODEL_PKL)
    print(f"[train] Saved feature pipeline → {FEATURE_PIPELINE_PKL}")
    print(f"[train] Saved best model → {BEST_MODEL_PKL}")

    # Log final production candidate to MLflow
    with mlflow.start_run(run_name=f"{best_name}_PRODUCTION"):
        mlflow.log_param("model", best_name)
        mlflow.log_param("trained_on", "full_train_data")
        mlflow.log_metric("cv_rmse_log", df_results.iloc[0]["mean_rmse_log"])
        mlflow.log_metric("cv_rmse_std", df_results.iloc[0]["std_rmse_log"])
        mlflow.sklearn.log_model(best_model_obj, "model")

    return df_results


if __name__ == "__main__":
    comparison = train_all_models(include_stacking=True)
    comparison.to_csv(DATA_PROCESSED / "model_comparison.csv", index=False)
    print("\n[train] Done. Model comparison saved.")
