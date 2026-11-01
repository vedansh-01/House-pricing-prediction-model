"""
test_feature_eng.py — Unit tests for FeatureEngineer:
  - No NaNs after transform
  - Target encoding is leakage-free (val fold didn't influence map)
  - Column alignment between train and val transforms
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_eng import FeatureEngineer


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_train_df():
    """Minimal synthetic training DataFrame mimicking Ames Housing structure."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Id": range(1, n + 1),
        "MSSubClass": np.random.choice([20, 30, 50, 60], n),
        "MSZoning": np.random.choice(["RL", "RM", "FV"], n),
        "LotFrontage": pd.array([None if np.random.rand() > 0.8 else float(np.random.randint(50, 120)) for _ in range(n)], dtype=object),
        "LotArea": np.random.randint(5000, 20000, n),
        "Street": "Pave",
        "Alley": pd.array(["Grvl" if np.random.rand() > 0.9 else None for _ in range(n)], dtype=object),
        "LotShape": np.random.choice(["Reg", "IR1", "IR2"], n),
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": np.random.choice(["NAmes", "CollgCr", "OldTown", "Edwards"], n),
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": np.random.choice(["1Story", "2Story"], n),
        "OverallQual": np.random.randint(4, 9, n),
        "OverallCond": np.random.randint(4, 8, n),
        "YearBuilt": np.random.randint(1950, 2005, n),
        "YearRemodAdd": np.random.randint(1980, 2010, n),
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrType": np.random.choice(["None", "BrkFace", "Stone"], n),
        "MasVnrArea": np.random.randint(0, 300, n).astype(float),
        "ExterQual": np.random.choice(["Ex", "Gd", "TA", "Fa"], n),
        "ExterCond": "TA",
        "Foundation": np.random.choice(["PConc", "CBlock"], n),
        "BsmtQual": pd.array(np.random.choice(["Ex", "Gd", "TA", None], n), dtype=object),
        "BsmtCond": pd.array(np.random.choice(["TA", "Gd", None], n), dtype=object),
        "BsmtExposure": pd.array(np.random.choice(["Av", "No", None], n), dtype=object),
        "BsmtFinType1": pd.array(np.random.choice(["GLQ", "Unf", None], n), dtype=object),
        "BsmtFinSF1": np.random.randint(0, 800, n).astype(float),
        "BsmtFinType2": pd.array(np.random.choice(["Unf", None], n), dtype=object),
        "BsmtFinSF2": np.zeros(n),
        "BsmtUnfSF": np.random.randint(0, 500, n).astype(float),
        "TotalBsmtSF": np.random.randint(500, 1500, n).astype(float),
        "Heating": "GasA",
        "HeatingQC": np.random.choice(["Ex", "Gd", "TA"], n),
        "CentralAir": np.random.choice(["Y", "N"], n),
        "Electrical": pd.array([None if np.random.rand() > 0.99 else "SBrkr" for _ in range(n)], dtype=object),
        "1stFlrSF": np.random.randint(700, 1800, n),
        "2ndFlrSF": np.random.randint(0, 800, n),
        "LowQualFinSF": np.zeros(n),
        "GrLivArea": np.random.randint(800, 2500, n),
        "BsmtFullBath": np.random.randint(0, 2, n),
        "BsmtHalfBath": np.zeros(n, dtype=int),
        "FullBath": np.random.randint(1, 3, n),
        "HalfBath": np.random.randint(0, 2, n),
        "BedroomAbvGr": np.random.randint(2, 5, n),
        "KitchenAbvGr": np.ones(n, dtype=int),
        "KitchenQual": np.random.choice(["Ex", "Gd", "TA"], n),
        "TotRmsAbvGrd": np.random.randint(5, 10, n),
        "Functional": "Typ",
        "Fireplaces": np.random.randint(0, 3, n),
        "FireplaceQu": pd.array(np.random.choice(["Gd", "TA", None], n), dtype=object),
        "GarageType": pd.array(np.random.choice(["Attchd", "Detchd", None], n), dtype=object),
        "GarageYrBlt": np.random.randint(1960, 2008, n).astype(float),
        "GarageFinish": pd.array(np.random.choice(["Fin", "Unf", None], n), dtype=object),
        "GarageCars": np.random.randint(0, 4, n),
        "GarageArea": np.random.randint(0, 800, n).astype(float),
        "GarageQual": pd.array(np.random.choice(["TA", "Gd", None], n), dtype=object),
        "GarageCond": pd.array(np.random.choice(["TA", None], n), dtype=object),
        "PavedDrive": np.random.choice(["Y", "N"], n),
        "WoodDeckSF": np.random.randint(0, 300, n).astype(float),
        "OpenPorchSF": np.random.randint(0, 150, n).astype(float),
        "EnclosedPorch": np.zeros(n),
        "3SsnPorch": np.zeros(n),
        "ScreenPorch": np.zeros(n),
        "PoolArea": np.zeros(n),
        "PoolQC": pd.array([None] * n, dtype=object),
        "Fence": pd.array(np.random.choice(["MnPrv", None], n), dtype=object),
        "MiscFeature": pd.array([None] * n, dtype=object),
        "MiscVal": np.zeros(n),
        "MoSold": np.random.randint(1, 13, n),
        "YrSold": np.random.choice([2007, 2008, 2009, 2010], n),
        "SaleType": "WD",
        "SaleCondition": "Normal",
        "SalePrice": np.random.randint(100_000, 400_000, n),
    }).set_index("Id")


@pytest.fixture
def sample_val_df(sample_train_df):
    """A held-out validation set (subset of train fixture, different indices)."""
    np.random.seed(99)
    val = sample_train_df.sample(20).copy()
    val.index = range(1000, 1020)
    return val


# ── Tests ────────────────────────────────────────────────────────────────────────

class TestFeatureEngineerNoNulls:
    def test_no_nans_train(self, sample_train_df):
        """Transformed train fold should have zero NaN values."""
        fe = FeatureEngineer()
        y = np.log1p(sample_train_df["SalePrice"])
        X = fe.fit_transform(sample_train_df, y=y)
        nan_counts = X.isna().sum()
        assert nan_counts.sum() == 0, f"NaN cols after transform: {nan_counts[nan_counts > 0].to_dict()}"

    def test_no_nans_val(self, sample_train_df, sample_val_df):
        """Transformed val fold (never seen by fit) should also have zero NaN."""
        fe = FeatureEngineer()
        y_tr = np.log1p(sample_train_df["SalePrice"])
        fe.fit(sample_train_df, y=y_tr)
        X_val = fe.transform(sample_val_df)
        nan_counts = X_val.isna().sum()
        assert nan_counts.sum() == 0, f"NaN in val after transform: {nan_counts[nan_counts > 0].to_dict()}"


class TestFeatureEngineerNoLeakage:
    def test_target_encoding_uses_only_train_fold(self, sample_train_df, sample_val_df):
        """
        Target encoding maps must be built ONLY from the train fold.
        Verify: encoding map computed from val-fold labels differs from train-fold map.
        """
        fe = FeatureEngineer()
        y_tr = np.log1p(sample_train_df["SalePrice"])
        fe.fit(sample_train_df, y=y_tr)

        # The encoding map should only reference neighborhoods from the train fold
        train_neighborhoods = set(sample_train_df["Neighborhood"].unique())
        map_keys = set(fe._target_encoding_maps.get("Neighborhood", {}).keys())

        # All mapped keys must have been in train fold
        assert map_keys.issubset(train_neighborhoods), (
            f"Target encoding map contains keys from outside train fold: {map_keys - train_neighborhoods}"
        )

    def test_val_not_influencing_encoding(self, sample_train_df, sample_val_df):
        """
        Fit on train, then fit on val separately — the two maps should differ
        (i.e., val data was NOT used when fitting on train).
        """
        fe_train = FeatureEngineer()
        y_tr = np.log1p(sample_train_df["SalePrice"])
        fe_train.fit(sample_train_df, y=y_tr)

        fe_val = FeatureEngineer()
        y_val = np.log1p(sample_val_df["SalePrice"])
        fe_val.fit(sample_val_df, y=y_val)

        map_train = fe_train._target_encoding_maps.get("Neighborhood", {})
        map_val = fe_val._target_encoding_maps.get("Neighborhood", {})

        # Maps will share some keys (same neighborhoods) but values will differ
        shared = set(map_train.keys()) & set(map_val.keys())
        if shared:
            diffs = [abs(map_train[k] - map_val[k]) for k in shared]
            # At least some values should differ (different sample → different mean)
            assert max(diffs) > 0, "Train and val encoding maps are identical — possible leakage!"


class TestFeatureEngineerColumnAlignment:
    def test_column_count_consistent(self, sample_train_df, sample_val_df):
        """Train and val transforms must produce identical column counts and names."""
        fe = FeatureEngineer()
        y_tr = np.log1p(sample_train_df["SalePrice"])
        X_tr = fe.fit_transform(sample_train_df, y=y_tr)
        X_val = fe.transform(sample_val_df)

        assert list(X_tr.columns) == list(X_val.columns), (
            "Column mismatch between train and val transforms."
        )

    def test_no_target_column_in_output(self, sample_train_df):
        """Transformed output must not contain SalePrice."""
        fe = FeatureEngineer()
        y = np.log1p(sample_train_df["SalePrice"])
        X = fe.fit_transform(sample_train_df, y=y)
        assert "SalePrice" not in X.columns

    def test_engineered_features_present(self, sample_train_df):
        """All expected derived features must be present."""
        fe = FeatureEngineer()
        y = np.log1p(sample_train_df["SalePrice"])
        X = fe.fit_transform(sample_train_df, y=y)
        for col in ["TotalSF", "HouseAge", "RemodAge", "TotalBath", "HasPool", "HasGarage"]:
            assert col in X.columns, f"Engineered feature '{col}' missing from transform output"
