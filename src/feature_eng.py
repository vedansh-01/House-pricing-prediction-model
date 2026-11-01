"""
feature_eng.py — FeatureEngineer class: all feature creation, imputation,
encoding, skew correction, and scaling in a single sklearn-compatible
fit/transform API.

LEAKAGE DISCIPLINE: fit() must only be called on the train fold.
transform() is then applied to both train and val folds (and inference inputs).
Target-encoding maps are learned inside fit() — never from val fold data.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.config import (
    ENGINEERED_COLS,
    MEANINGFUL_NAN_COLS,
    OHE_COLS,
    QUALITY_COLS,
    QUALITY_MAP,
    RANDOM_STATE,
    SKEW_THRESHOLD,
    TARGET_COL,
    TARGET_ENCODE_COLS,
)

warnings.filterwarnings("ignore", category=FutureWarning)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    End-to-end feature engineering pipeline.

    Steps (in order):
      1. Meaningful-NaN fill (categorical "absent" → "None")
      2. True-missing imputation (LotFrontage by Neighborhood median, Electrical mode)
      3. Derived feature creation (TotalSF, HouseAge, etc.)
      4. Ordinal quality encoding
      5. One-hot encoding (low-cardinality nominals)
      6. Target encoding (Neighborhood) — fit on train fold only
      7. Skew correction on numeric features
      8. (Optional) StandardScaler — enabled for linear models
    """

    def __init__(self, use_scaler: bool = False):
        self.use_scaler = use_scaler

        # Learned parameters (set during fit)
        self._lot_frontage_medians: Optional[pd.Series] = None
        self._electrical_mode: Optional[str] = None
        self._target_encoding_maps: dict[str, dict] = {}
        self._target_global_mean: float = 0.0
        self._ohe_columns: list[str] = []          # actual OHE output column names
        self._skew_cols: list[str] = []             # numeric cols above skew threshold
        self._feature_cols: list[str] = []          # final ordered column list
        self._numeric_cols_fit: list[str] = []      # numeric cols seen at fit time
        self._scaler: Optional[StandardScaler] = None
        self._fitted = False

    # ── Public API ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """
        Learn all imputation/encoding parameters from train fold df.
        y must be log1p(SalePrice) when provided (used for target encoding).
        """
        df = df.copy()

        # Drop target column immediately — never let it appear in feature set
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])

        # Step 1 — record numeric columns present in training data
        self._numeric_cols_fit = [
            c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET_COL
        ]

        # Step 2 — fill meaningful NaN cols
        df = self._fill_meaningful_nans(df)

        # Step 3 — learn imputation params
        self._lot_frontage_medians = (
            df.groupby("Neighborhood")["LotFrontage"].median()
        )
        if "Electrical" in df.columns:
            self._electrical_mode = df["Electrical"].mode()[0]

        # Step 3 — create engineered features (needed for skew detection)
        df = self._create_features(df)

        # Step 4 — ordinal encoding
        df = self._encode_ordinal(df)

        # Step 5 — learn OHE structure from train fold
        df_ohe = self._ohe_fit_transform(df)

        # Step 6 — learn target encoding from train fold
        if y is not None:
            log_target = y if y.name != TARGET_COL else np.log1p(y)
            self._target_global_mean = float(log_target.mean())
            for col in TARGET_ENCODE_COLS:
                if col in df.columns:
                    self._target_encoding_maps[col] = (
                        df[col]
                        .map(lambda v, c=col, logy=log_target, dfcol=df[col]:
                             logy[dfcol == v].mean())
                        .groupby(df[col])
                        .first()
                        .to_dict()
                    )
                    # simpler, cleaner approach:
                    tmp = pd.DataFrame({"cat": df[col], "target": log_target})
                    self._target_encoding_maps[col] = (
                        tmp.groupby("cat")["target"].mean().to_dict()
                    )
                    df_ohe[col] = df[col].map(self._target_encoding_maps[col]).fillna(
                        self._target_global_mean
                    )
        else:
            # no target available (test set at inference): encode to global mean
            for col in TARGET_ENCODE_COLS:
                if col in df_ohe.columns or col in df.columns:
                    df_ohe[col] = self._target_global_mean

        # Step 7 — detect skewed numeric cols
        numeric_cols = df_ohe.select_dtypes(include=[np.number]).columns.tolist()
        skewness = df_ohe[numeric_cols].apply(lambda x: skew(x.dropna())).abs()
        self._skew_cols = skewness[skewness > SKEW_THRESHOLD].index.tolist()
        # Don't skew-transform binary columns (HasPool, HasGarage)
        binary_cols = [c for c in self._skew_cols if df_ohe[c].nunique() <= 2]
        self._skew_cols = [c for c in self._skew_cols if c not in binary_cols]

        df_ohe = self._apply_skew_transform(df_ohe)

        # Step 8 — scaler
        self._feature_cols = df_ohe.columns.tolist()
        if self.use_scaler:
            self._scaler = StandardScaler()
            self._scaler.fit(df_ohe)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all learned transforms to new data. Returns clean DataFrame."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        df = df.copy()

        # Drop target if present (inference case)
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])

        df = self._fill_meaningful_nans(df)
        df = self._impute_true_missing(df)
        df = self._create_features(df)
        df = self._encode_ordinal(df)
        df = self._ohe_transform(df)

        # Target encoding (apply learned maps)
        for col in TARGET_ENCODE_COLS:
            if col in df.columns:
                df[col] = df[col].map(self._target_encoding_maps.get(col, {})).fillna(
                    self._target_global_mean
                )
            elif col not in df.columns:
                df[col] = self._target_global_mean

        df = self._apply_skew_transform(df)

        # Align columns to training schema (add missing cols as 0, drop extras)
        missing = [col for col in self._feature_cols if col not in df.columns]
        if missing:
            missing_df = pd.DataFrame(0, index=df.index, columns=missing)
            df = pd.concat([df, missing_df], axis=1)
        df = df[self._feature_cols]

        if self.use_scaler and self._scaler is not None:
            arr = self._scaler.transform(df)
            df = pd.DataFrame(arr, index=df.index, columns=df.columns)

        return df

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _fill_meaningful_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN in 'absent feature' columns with the string 'None'."""
        for col in MEANINGFUL_NAN_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("None")
        return df

    def _impute_true_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute truly-missing values using parameters learned on train fold."""
        # Ensure any known numeric columns that came in as object/None are converted to float/int
        if hasattr(self, "_numeric_cols_fit") and self._numeric_cols_fit:
            for col in self._numeric_cols_fit:
                if col in df.columns and col != "LotFrontage":
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # LotFrontage: fill with neighborhood median learned from train fold
        if "LotFrontage" in df.columns and self._lot_frontage_medians is not None:
            mask = df["LotFrontage"].isna()
            df.loc[mask, "LotFrontage"] = df.loc[mask, "Neighborhood"].map(
                self._lot_frontage_medians
            )
            # fallback for neighborhoods not seen at fit time
            df["LotFrontage"] = pd.to_numeric(df["LotFrontage"], errors="coerce").fillna(
                float(self._lot_frontage_medians.median())
            )

        # Electrical
        if "Electrical" in df.columns and self._electrical_mode is not None:
            df["Electrical"] = df["Electrical"].fillna(self._electrical_mode)

        # Any remaining numeric NaNs → 0 (safe default for tree models)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)

        # Any remaining categorical NaNs → "Unknown"
        cat_cols = df.select_dtypes(include=["object"]).columns
        df[cat_cols] = df[cat_cols].fillna("Unknown")

        return df

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features."""
        df["TotalSF"] = (
            df.get("TotalBsmtSF", 0)
            + df.get("1stFlrSF", 0)
            + df.get("2ndFlrSF", 0)
        )
        df["HouseAge"] = df.get("YrSold", 2010) - df.get("YearBuilt", 2000)
        df["RemodAge"] = df.get("YrSold", 2010) - df.get("YearRemodAdd", 2000)
        df["TotalBath"] = (
            df.get("FullBath", 0)
            + 0.5 * df.get("HalfBath", 0)
            + df.get("BsmtFullBath", 0)
            + 0.5 * df.get("BsmtHalfBath", 0)
        )
        df["HasPool"] = (df.get("PoolArea", pd.Series(0, index=df.index)) > 0).astype(int)
        df["HasGarage"] = (df.get("GarageArea", pd.Series(0, index=df.index)) > 0).astype(int)
        return df

    def _encode_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply manual ordinal map to quality columns."""
        for col in QUALITY_COLS:
            if col in df.columns:
                df[col] = df[col].map(QUALITY_MAP).fillna(0).astype(int)
        return df

    def _ohe_fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode during fit — records column schema."""
        cols_present = [c for c in OHE_COLS if c in df.columns]
        df_encoded = pd.get_dummies(df, columns=cols_present, drop_first=False, dtype=int)
        # Store OHE output columns for alignment during transform
        self._ohe_columns = df_encoded.columns.tolist()
        return df_encoded

    def _ohe_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode during transform — aligns to fit-time schema."""
        cols_present = [c for c in OHE_COLS if c in df.columns]
        df_encoded = pd.get_dummies(df, columns=cols_present, drop_first=False, dtype=int)
        # Align: add missing cols as 0, keep only fit-time cols
        if self._ohe_columns:
            missing = [c for c in self._ohe_columns if c not in df_encoded.columns]
            if missing:
                missing_df = pd.DataFrame(0, index=df_encoded.index, columns=missing)
                df_encoded = pd.concat([df_encoded, missing_df], axis=1)
            # Only keep columns seen at fit time (plus non-OHE cols)
            available = [c for c in self._ohe_columns if c in df_encoded.columns]
            df_encoded = df_encoded[available]
        return df_encoded

    def _apply_skew_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log1p transform numeric columns with high skewness."""
        for col in self._skew_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))
        return df

    @property
    def feature_names_(self) -> list[str]:
        """Return final feature column names after full transform."""
        return self._feature_cols
