"""
test_predict.py — Unit tests for the predict module.
Requires model files to be present (post-training).
"""

import pytest

from api import model_loader

MODEL_READY = model_loader.is_ready()

SAMPLE_INPUT = {
    "MSSubClass": 60,
    "MSZoning": "RL",
    "LotFrontage": 65.0,
    "LotArea": 8450,
    "Street": "Pave",
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Neighborhood": "CollgCr",
    "Condition1": "Norm",
    "Condition2": "Norm",
    "BldgType": "1Fam",
    "HouseStyle": "2Story",
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2003,
    "YearRemodAdd": 2003,
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "BrkFace",
    "MasVnrArea": 162.0,
    "ExterQual": "Gd",
    "ExterCond": "TA",
    "Foundation": "PConc",
    "TotalBsmtSF": 856.0,
    "BsmtFinSF1": 706.0,
    "BsmtFinSF2": 0.0,
    "BsmtUnfSF": 150.0,
    "BsmtFullBath": 1,
    "BsmtHalfBath": 0,
    "Heating": "GasA",
    "HeatingQC": "Ex",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "1stFlrSF": 856,
    "2ndFlrSF": 854,
    "LowQualFinSF": 0,
    "GrLivArea": 1710,
    "FullBath": 2,
    "HalfBath": 1,
    "BedroomAbvGr": 3,
    "KitchenAbvGr": 1,
    "KitchenQual": "Gd",
    "TotRmsAbvGrd": 8,
    "Functional": "Typ",
    "Fireplaces": 1,
    "GarageCars": 2,
    "GarageArea": 548.0,
    "PavedDrive": "Y",
    "WoodDeckSF": 0.0,
    "OpenPorchSF": 61.0,
    "EnclosedPorch": 0.0,
    "3SsnPorch": 0.0,
    "ScreenPorch": 0.0,
    "PoolArea": 0.0,
    "MiscVal": 0.0,
    "MoSold": 2,
    "YrSold": 2008,
    "SaleType": "WD",
    "SaleCondition": "Normal",
}


@pytest.mark.skipif(not MODEL_READY, reason="Model not trained yet — run src/train.py first")
def test_predict_returns_float():
    price = model_loader.predict(SAMPLE_INPUT)
    assert isinstance(price, float), f"Expected float, got {type(price)}"


@pytest.mark.skipif(not MODEL_READY, reason="Model not trained yet")
def test_predict_positive_price():
    price = model_loader.predict(SAMPLE_INPUT)
    assert price > 0, f"Price should be positive, got {price}"


@pytest.mark.skipif(not MODEL_READY, reason="Model not trained yet")
def test_predict_batch_returns_list():
    prices = model_loader.predict_batch([SAMPLE_INPUT, SAMPLE_INPUT])
    assert isinstance(prices, list)
    assert len(prices) == 2
    assert all(isinstance(p, float) and p > 0 for p in prices)


@pytest.mark.skipif(not MODEL_READY, reason="Model not trained yet")
def test_higher_quality_predicts_higher_price():
    """A higher quality house should predict a higher price (directional sanity check)."""
    low_qual = {**SAMPLE_INPUT, "OverallQual": 3, "GrLivArea": 800}
    high_qual = {**SAMPLE_INPUT, "OverallQual": 9, "GrLivArea": 2500}
    price_low = model_loader.predict(low_qual)
    price_high = model_loader.predict(high_qual)
    assert price_high > price_low, (
        f"Higher quality house should be more expensive: {price_high:.0f} vs {price_low:.0f}"
    )
