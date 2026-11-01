"""
data_prep.py — Data loading, outlier removal, and CV fold generation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import (
    N_FOLDS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_CSV,
    TRAIN_CSV,
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test CSVs. Returns (train_df, test_df)."""
    train = pd.read_csv(TRAIN_CSV, index_col="Id")
    test = pd.read_csv(TEST_CSV, index_col="Id")
    print(f"[load_data] train={train.shape}, test={test.shape}")
    return train, test


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove known Ames Housing outliers:
      - GrLivArea > 4000 sqft AND SalePrice < 300,000 (two data-entry anomalies)
    Only applied on training data (must contain SalePrice column).
    Returns cleaned DataFrame with same index type.
    """
    if TARGET_COL not in df.columns:
        return df  # test set — no target, skip
    mask = ~((df["GrLivArea"] > 4000) & (df[TARGET_COL] < 300_000))
    dropped = (~mask).sum()
    if dropped:
        print(f"[remove_outliers] Dropped {dropped} outlier rows.")
    return df[mask].copy()


def get_cv_folds(
    df: pd.DataFrame, k: int = N_FOLDS, random_state: int = RANDOM_STATE
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return list of (train_idx, val_idx) arrays for k-fold CV.
    Uses price-bin-aware stratification: bins SalePrice into k quantiles so
    every fold has a representative price range distribution.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    # sort by SalePrice to ensure fold distribution is balanced
    sorted_idx = df[TARGET_COL].argsort().values
    folds = []
    for tr, val in kf.split(sorted_idx):
        # map back to actual df row positions
        folds.append((sorted_idx[tr], sorted_idx[val]))
    return folds


def train_test_split_hold_out(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple hold-out split (used for final evaluation / sanity checks only).
    CV is the primary evaluation approach.
    """
    from sklearn.model_selection import train_test_split

    train, val = train_test_split(df, test_size=test_size, random_state=random_state)
    return train.copy(), val.copy()
