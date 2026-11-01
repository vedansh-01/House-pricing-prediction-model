"""
config.py — Central configuration: paths, constants, feature category lists.
All other modules import from here; change paths/feature lists only here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MONITORING_DIR = BASE_DIR / "monitoring"

TRAIN_CSV = DATA_RAW / "train.csv"
TEST_CSV = DATA_RAW / "test.csv"
FEATURE_PIPELINE_PKL = DATA_PROCESSED / "feature_pipeline.pkl"
BEST_MODEL_PKL = MODELS_DIR / "best_model.pkl"
REFERENCE_DATA_CSV = MONITORING_DIR / "reference_data.csv"

# ── Target ─────────────────────────────────────────────────────────────────────
TARGET_COL = "SalePrice"

# ── Outlier indices to remove (identified during EDA) ─────────────────────────
# GrLivArea > 4000 sqft with SalePrice < 300k are the two documented Ames outliers
OUTLIER_INDICES: list[int] = []  # populated at runtime by data_prep.remove_outliers()

# ── Quality ordinal mapping (None → Ex scale) ──────────────────────────────────
QUALITY_MAP: dict[str, int] = {
    "None": 0,
    "Po": 1,
    "Fa": 2,
    "TA": 3,
    "Gd": 4,
    "Ex": 5,
}

QUALITY_COLS: list[str] = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual",
    "GarageCond", "PoolQC",
]

# ── Columns whose NaN means "feature absent" (not truly missing) ───────────────
MEANINGFUL_NAN_COLS: list[str] = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature",
]

# ── Nominal low-cardinality columns → one-hot encoded ─────────────────────────
OHE_COLS: list[str] = [
    "MSZoning", "Street", "LotShape", "LandContour", "Utilities",
    "LotConfig", "LandSlope", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
    "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation",
    "Heating", "CentralAir", "Electrical", "Functional",
    "GarageType", "GarageFinish", "PavedDrive", "SaleType",
    "SaleCondition", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "Fence", "MiscFeature", "Alley",
]

# ── Target-encoded column (high cardinality) ───────────────────────────────────
TARGET_ENCODE_COLS: list[str] = ["Neighborhood"]

# ── Numeric columns created during feature engineering ────────────────────────
ENGINEERED_COLS: list[str] = [
    "TotalSF", "HouseAge", "RemodAge", "TotalBath", "HasPool", "HasGarage",
]

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# ── Cross-validation ───────────────────────────────────────────────────────────
N_FOLDS = 5
RANDOM_STATE = 42

# ── Skew threshold for log-transform ──────────────────────────────────────────
SKEW_THRESHOLD = 0.75

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "house_price_prediction"
MLFLOW_TRACKING_DIR = BASE_DIR / "mlruns"
MLFLOW_TRACKING_URI = MLFLOW_TRACKING_DIR.as_uri()
