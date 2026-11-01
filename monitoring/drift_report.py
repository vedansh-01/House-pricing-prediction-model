"""
drift_report.py — Evidently AI data drift monitoring.

Usage:
    python -m monitoring.drift_report

Produces:
    - monitoring/drift_report.html  (Evidently HTML report)
    - monitoring/drift_summary.json (machine-readable drift status per feature)
    - Console output: which features drifted and retrain recommendation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path so src.* imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BEST_MODEL_PKL, FEATURE_PIPELINE_PKL, REFERENCE_DATA_CSV, TARGET_COL
from src.data_prep import load_data, remove_outliers

MONITORING_DIR = Path(__file__).resolve().parent
DRIFT_REPORT_HTML = MONITORING_DIR / "drift_report.html"
DRIFT_SUMMARY_JSON = MONITORING_DIR / "drift_summary.json"


def save_reference_data() -> None:
    """
    Save a snapshot of training feature distribution to reference_data.csv.
    Run once after training to establish the baseline.
    """
    train_df, _ = load_data()
    train_df = remove_outliers(train_df)
    fe = joblib.load(FEATURE_PIPELINE_PKL)
    X_ref = fe.transform(train_df)

    REFERENCE_DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    X_ref.to_csv(REFERENCE_DATA_CSV, index=False)
    print(f"[drift] Reference data saved → {REFERENCE_DATA_CSV}  shape={X_ref.shape}")


def create_simulated_drift_batch(reference_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic 'new data' batch by perturbing the reference distribution.
    Shifts several numeric columns to simulate real-world data drift.
    """
    rng = np.random.RandomState(seed)
    new_df = reference_df.copy()

    # Perturb numeric features to simulate drift
    numeric_cols = new_df.select_dtypes(include=[np.number]).columns

    # Drift top correlated features by shifting their distribution
    drift_cols = ["GrLivArea", "TotalBsmtSF", "LotArea", "GarageArea",
                  "TotalSF", "OverallQual"]
    drift_cols = [c for c in drift_cols if c in new_df.columns]

    for col in drift_cols[:3]:  # drift a subset for realism
        shift = new_df[col].std() * 0.8
        new_df[col] = new_df[col] + shift + rng.normal(0, shift * 0.2, len(new_df))

    # Add extra variance to some columns
    for col in numeric_cols[:5]:
        if col not in drift_cols:
            new_df[col] = new_df[col] * rng.uniform(1.1, 1.4, len(new_df))

    return new_df.clip(lower=0)


def run_drift_report(
    use_simulated_drift: bool = True,
    new_data_csv: str | None = None,
) -> dict:
    """
    Run Evidently AI DataDrift report comparing reference vs new batch.

    Args:
        use_simulated_drift: if True, generate perturbed data for testing
        new_data_csv: path to real new-batch CSV (overrides simulation)

    Returns:
        drift_summary dict with per-feature drift status
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset

    # Load reference data
    if not REFERENCE_DATA_CSV.exists():
        print("[drift] Reference data not found — generating it now ...")
        save_reference_data()
    reference_df = pd.read_csv(REFERENCE_DATA_CSV)
    print(f"[drift] Reference data loaded: {reference_df.shape}")

    # Load new data
    if new_data_csv and Path(new_data_csv).exists():
        new_df = pd.read_csv(new_data_csv)
        print(f"[drift] New data loaded from {new_data_csv}: {new_df.shape}")
    elif use_simulated_drift:
        new_df = create_simulated_drift_batch(reference_df)
        print(f"[drift] Using simulated drifted batch: {new_df.shape}")
    else:
        print("[drift] No new data provided and simulation disabled. Exiting.")
        return {}

    # ── Evidently Report ────────────────────────────────────────────────────────
    # Use a representative subset of columns to keep report manageable
    feature_cols = reference_df.columns[:50].tolist()
    ref_subset = reference_df[feature_cols].copy()
    new_subset = new_df[feature_cols].copy()

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref_subset, current_data=new_subset)

    # Save HTML report
    snapshot.save_html(str(DRIFT_REPORT_HTML))
    print(f"[drift] HTML report saved → {DRIFT_REPORT_HTML}")

    # ── Extract drift summary ───────────────────────────────────────────────────
    result_dict = snapshot.dict()
    drift_summary = {"drifted_features": [], "stable_features": [], "retrain_recommended": False}

    try:
        metrics_data = result_dict.get("metrics", [])
        for metric in metrics_data:
            name = metric.get("metric_name", "")
            val = metric.get("value")
            if "DriftedColumnsCount" in name and isinstance(val, dict):
                drift_summary["n_drifted_columns"] = int(val.get("count", 0))
                drift_summary["share_drifted"] = float(val.get("share", 0.0))
                if drift_summary["share_drifted"] > 0.20:
                    drift_summary["retrain_recommended"] = True
            elif "ValueDrift" in name:
                # e.g. ValueDrift(column=GrLivArea,...)
                col_name = name.split("column=")[1].split(",")[0] if "column=" in name else "feature"
                p_val = float(val) if val is not None else 1.0
                if p_val < 0.05:
                    drift_summary["drifted_features"].append(col_name)
                else:
                    drift_summary["stable_features"].append(col_name)
    except Exception as e:
        print(f"[drift] Could not parse drift summary: {e}")

    # Save JSON summary
    with open(DRIFT_SUMMARY_JSON, "w") as f:
        json.dump(drift_summary, f, indent=2)

    # ── Console summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  DATA DRIFT SUMMARY")
    print("=" * 55)
    print(f"  Drifted features ({len(drift_summary['drifted_features'])}):")
    for f in drift_summary["drifted_features"][:10]:
        print(f"    ✗  {f}")
    print(f"\n  Share drifted: {drift_summary.get('share_drifted', 0):.1%}")
    rec = drift_summary["retrain_recommended"]
    print(f"  Retrain recommended: {'YES ⚠' if rec else 'No ✓'}")
    print(f"\n  Full report → {DRIFT_REPORT_HTML}")
    print(f"  JSON summary → {DRIFT_SUMMARY_JSON}")
    print("=" * 55)

    return drift_summary


if __name__ == "__main__":
    run_drift_report(use_simulated_drift=True)
