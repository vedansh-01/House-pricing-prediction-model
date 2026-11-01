"""
tune.py — Optuna hyperparameter tuning for best tree model (XGBoost / LightGBM).

Usage:
    python -m src.tune [--model xgboost|lightgbm] [--trials 100]

Produces:
    - Updated best_model.pkl  (tuned model, refitted on full data)
    - Optuna study visualization (saved as HTML if plotly available)
    - MLflow run: "XGBoost_TUNED" or "LightGBM_TUNED"
"""

from __future__ import annotations

import argparse
import warnings

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.config import (
    BEST_MODEL_PKL,
    FEATURE_PIPELINE_PKL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_FOLDS,
    RANDOM_STATE,
    TARGET_COL,
)
from src.data_prep import get_cv_folds, load_data, remove_outliers
from src.evaluate import rmse_log
from src.feature_eng import FeatureEngineer

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ── Search space definitions ───────────────────────────────────────────────────

def xgboost_objective(trial: optuna.Trial, df_train: pd.DataFrame, folds) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    }
    return _cv_objective(XGBRegressor(**params), df_train, folds)


def lightgbm_objective(trial: optuna.Trial, df_train: pd.DataFrame, folds) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }
    return _cv_objective(LGBMRegressor(**params), df_train, folds)


def _cv_objective(model, df_train: pd.DataFrame, folds) -> float:
    """Evaluate a model configuration via CV — returns mean RMSE log."""
    fold_rmse = []
    for tr_idx, val_idx in folds:
        fold_tr = df_train.iloc[tr_idx].copy()
        fold_val = df_train.iloc[val_idx].copy()
        y_tr = np.log1p(fold_tr[TARGET_COL])
        y_val = np.log1p(fold_val[TARGET_COL])

        fe = FeatureEngineer(use_scaler=False)
        X_tr = fe.fit_transform(fold_tr, y=y_tr)
        X_val = fe.transform(fold_val)

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        fold_rmse.append(rmse_log(y_val.values, y_pred))
    return float(np.mean(fold_rmse))


# ── Main tuning function ───────────────────────────────────────────────────────

def tune(model_type: str = "lightgbm", n_trials: int = 100) -> None:
    """
    Run Optuna hyperparameter search for the chosen model type.
    After finding best params, refit on full training data and serialize.
    """
    assert model_type in ("xgboost", "lightgbm"), "model_type must be 'xgboost' or 'lightgbm'"

    print(f"\n[tune] Starting {model_type} tuning — {n_trials} trials ...")
    train_df, _ = load_data()
    train_df = remove_outliers(train_df)
    folds = get_cv_folds(train_df, k=N_FOLDS)

    objective_fn = xgboost_objective if model_type == "xgboost" else lightgbm_objective
    study = optuna.create_study(direction="minimize", study_name=f"{model_type}_tuning")
    study.optimize(
        lambda trial: objective_fn(trial, train_df, folds),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_rmse = study.best_value
    print(f"\n[tune] Best RMSE: {best_rmse:.4f}")
    print(f"[tune] Best params: {best_params}")

    # ── Save Optuna importance visualization ────────────────────────────────────
    try:
        import plotly
        fig_importance = optuna.visualization.plot_param_importances(study)
        fig_importance.write_html(str(BEST_MODEL_PKL.parent / f"{model_type}_param_importance.html"))
        fig_contour = optuna.visualization.plot_optimization_history(study)
        fig_contour.write_html(str(BEST_MODEL_PKL.parent / f"{model_type}_optimization_history.html"))
        print("[tune] Saved Optuna visualizations as HTML.")
    except Exception:
        print("[tune] Plotly not available — skipping Optuna visualizations.")

    # ── Refit best model on full training data ──────────────────────────────────
    if model_type == "xgboost":
        best_model = XGBRegressor(
            **{k: v for k, v in best_params.items()},
            objective="reg:squarederror", n_jobs=-1,
            random_state=RANDOM_STATE, verbosity=0,
        )
    else:
        best_model = LGBMRegressor(
            **{k: v for k, v in best_params.items()},
            n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
        )

    y_full = np.log1p(train_df[TARGET_COL])
    fe_full = FeatureEngineer(use_scaler=False)
    X_full = fe_full.fit_transform(train_df, y=y_full)
    best_model.fit(X_full, y_full)

    # Serialize
    joblib.dump(fe_full, FEATURE_PIPELINE_PKL)
    joblib.dump(best_model, BEST_MODEL_PKL)
    print(f"[tune] Tuned model saved → {BEST_MODEL_PKL}")

    # ── Log to MLflow ───────────────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{model_type.upper()}_TUNED"):
        mlflow.log_param("model", model_type)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_params(best_params)
        try:
            mlflow.sklearn.log_model(best_model, "model", serialization_format="pickle")
        except Exception:
            try:
                import mlflow.xgboost
                mlflow.xgboost.log_model(best_model, "model")
            except Exception as e:
                print(f"[tune] Note: MLflow model artifact logging skipped ({e}). Metrics/params logged.")
    print(f"[tune] Logged tuned run to MLflow experiment '{MLFLOW_EXPERIMENT_NAME}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--model", default="lightgbm", choices=["xgboost", "lightgbm"])
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    tune(model_type=args.model, n_trials=args.trials)
