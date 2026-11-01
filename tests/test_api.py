"""
test_api.py — Integration tests for the FastAPI prediction endpoints.
Uses TestClient (no running server required).

Note: Tests that touch /predict require model files.
Tests are skipped gracefully if model files are missing (pre-training).
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api import model_loader

client = TestClient(app)

MODEL_READY = model_loader.is_ready()


# ── Health endpoint ─────────────────────────────────────────────────────────────

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_schema():
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "pipeline_loaded" in data
    assert "model_version" in data


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert "docs" in body


# ── Predict endpoint (requires model) ──────────────────────────────────────────

SAMPLE_PAYLOAD = {
    "MSSubClass": 60,
    "MSZoning": "RL",
    "LotArea": 9600,
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
    "BsmtFinSF1": 706.0,
    "BsmtFinSF2": 0.0,
    "BsmtUnfSF": 150.0,
    "TotalBsmtSF": 856.0,
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


@pytest.mark.skipif(not MODEL_READY, reason="Model not trained yet")
def test_predict_returns_200():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200


@pytest.mark.skipif(not MODEL_READY, reason="Model not trained yet")
def test_predict_response_has_price():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)
    assert data["predicted_price"] > 0
    assert data["currency"] == "USD"


@pytest.mark.skipif(not MODEL_READY, reason="Model not trained yet")
def test_predict_price_in_reasonable_range():
    """Predicted price should be in a plausible range for Ames Housing."""
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    price = response.json()["predicted_price"]
    assert 50_000 <= price <= 1_000_000, f"Price {price} outside plausible range"


def test_predict_returns_422_on_invalid_type():
    """Sending wrong type for OverallQual should trigger 422 validation error."""
    bad_payload = {**SAMPLE_PAYLOAD, "OverallQual": "not_an_int"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_returns_422_on_out_of_range():
    """OverallQual must be 1-10; 15 should fail validation."""
    bad_payload = {**SAMPLE_PAYLOAD, "OverallQual": 15}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


@pytest.mark.skipif(not MODEL_READY, reason="Model not trained yet")
def test_batch_predict():
    batch_payload = {"houses": [SAMPLE_PAYLOAD, SAMPLE_PAYLOAD]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert data["count"] == 2
