"""
app.py — Interactive Streamlit Web UI for House Price Prediction.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
from src.predict import predict
from src.config import BEST_MODEL_PKL, FEATURE_PIPELINE_PKL

st.set_page_config(
    page_title="Ames House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏡 Ames House Price Prediction System")
st.markdown(
    "Predict the market value of residential properties in Ames, Iowa using our "
    "**leakage-free machine learning pipeline** (Optuna-tuned XGBoost / Regularized Regression)."
)

# ── Preset Selection ──────────────────────────────────────────────────────────
PRESETS = {
    "Typical Suburban Home (Mid-tier)": {
        "OverallQual": 6,
        "GrLivArea": 1600,
        "TotalBsmtSF": 950.0,
        "YearBuilt": 1995,
        "YearRemodAdd": 2005,
        "GarageCars": 2,
        "GarageArea": 480.0,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "TotRmsAbvGrd": 7,
        "Neighborhood": "CollgCr",
        "ExterQual": "Gd",
        "KitchenQual": "Gd",
        "BsmtQual": "Gd",
        "HeatingQC": "Ex",
        "Fireplaces": 1,
        "LotArea": 9500,
        "1stFlrSF": 950,
        "2ndFlrSF": 650,
    },
    "Luxury High-End Estate": {
        "OverallQual": 9,
        "GrLivArea": 3200,
        "TotalBsmtSF": 1800.0,
        "YearBuilt": 2008,
        "YearRemodAdd": 2010,
        "GarageCars": 3,
        "GarageArea": 850.0,
        "FullBath": 3,
        "HalfBath": 1,
        "BedroomAbvGr": 4,
        "TotRmsAbvGrd": 10,
        "Neighborhood": "StoneBr",
        "ExterQual": "Ex",
        "KitchenQual": "Ex",
        "BsmtQual": "Ex",
        "HeatingQC": "Ex",
        "Fireplaces": 2,
        "LotArea": 16000,
        "1stFlrSF": 1800,
        "2ndFlrSF": 1400,
    },
    "Starter / Budget Home": {
        "OverallQual": 4,
        "GrLivArea": 980,
        "TotalBsmtSF": 750.0,
        "YearBuilt": 1955,
        "YearRemodAdd": 1960,
        "GarageCars": 1,
        "GarageArea": 260.0,
        "FullBath": 1,
        "HalfBath": 0,
        "BedroomAbvGr": 2,
        "TotRmsAbvGrd": 5,
        "Neighborhood": "OldTown",
        "ExterQual": "TA",
        "KitchenQual": "TA",
        "BsmtQual": "TA",
        "HeatingQC": "TA",
        "Fireplaces": 0,
        "LotArea": 6000,
        "1stFlrSF": 980,
        "2ndFlrSF": 0,
    },
}

st.sidebar.header("🎯 Preset Profiles")
preset_name = st.sidebar.selectbox("Load a sample house:", list(PRESETS.keys()))
default_vals = PRESETS[preset_name]

# ── Feature Inputs Form ───────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📐 Living Area & Size")
    gr_liv_area = st.slider("Above-Ground Living Area (sq ft)", 500, 4500, int(default_vals["GrLivArea"]), step=25)
    total_bsmt_sf = st.slider("Basement Area (sq ft)", 0, 3000, int(default_vals["TotalBsmtSF"]), step=25)
    lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=60000, value=int(default_vals["LotArea"]), step=500)
    
    # Calculate 1st and 2nd floor approximation
    first_flr_sf = min(gr_liv_area, int(total_bsmt_sf if total_bsmt_sf > 0 else gr_liv_area))
    second_flr_sf = max(0, gr_liv_area - first_flr_sf)

with col2:
    st.subheader("⭐ Quality & Age")
    overall_qual = st.slider("Overall Quality (1=Very Poor, 10=Very Excellent)", 1, 10, int(default_vals["OverallQual"]))
    year_built = st.slider("Year Built", 1900, 2010, int(default_vals["YearBuilt"]))
    year_remod = st.slider("Year Remodeled / Added", 1950, 2010, max(int(default_vals["YearRemodAdd"]), year_built))
    
    qual_options = ["Po", "Fa", "TA", "Gd", "Ex"]
    exter_qual = st.selectbox("Exterior Quality", qual_options, index=qual_options.index(default_vals["ExterQual"]))
    kitchen_qual = st.selectbox("Kitchen Quality", qual_options, index=qual_options.index(default_vals["KitchenQual"]))

with col3:
    st.subheader("🚗 Rooms, Garage & Location")
    neighborhoods = [
        "CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NWAmes", 
        "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes", "SawyerW", "IDOTRR", 
        "MeadowV", "Edwards", "Timber", "Gilbert", "StoneBr", "ClearCr", "NPkVill", 
        "Blmngtn", "BrDale", "SWISU", "Blueste"
    ]
    neighborhood = st.selectbox("Neighborhood", sorted(neighborhoods), index=sorted(neighborhoods).index(default_vals["Neighborhood"]))
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        bedrooms = st.number_input("Bedrooms (Above Grade)", 1, 6, int(default_vals["BedroomAbvGr"]))
        full_bath = st.number_input("Full Bathrooms", 1, 4, int(default_vals["FullBath"]))
    with col_c2:
        tot_rooms = st.number_input("Total Rooms", 3, 14, int(default_vals["TotRmsAbvGrd"]))
        garage_cars = st.number_input("Garage Cars", 0, 4, int(default_vals["GarageCars"]))
    
    garage_area = garage_cars * 240.0
    fireplaces = st.number_input("Fireplaces", 0, 3, int(default_vals["Fireplaces"]))

st.divider()

# ── Inference Execution ───────────────────────────────────────────────────────
house_data = {
    "MSSubClass": 20,
    "MSZoning": "RL",
    "LotFrontage": 70.0,
    "LotArea": lot_area,
    "Street": "Pave",
    "Alley": None,
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Neighborhood": neighborhood,
    "Condition1": "Norm",
    "Condition2": "Norm",
    "BldgType": "1Fam",
    "HouseStyle": "1Story" if second_flr_sf == 0 else "2Story",
    "OverallQual": overall_qual,
    "OverallCond": 5,
    "YearBuilt": year_built,
    "YearRemodAdd": year_remod,
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "None",
    "MasVnrArea": 0.0,
    "ExterQual": exter_qual,
    "ExterCond": "TA",
    "Foundation": "PConc",
    "BsmtQual": default_vals.get("BsmtQual", "TA"),
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "GLQ" if total_bsmt_sf > 0 else "None",
    "BsmtFinSF1": total_bsmt_sf * 0.6 if total_bsmt_sf > 0 else 0.0,
    "BsmtFinType2": "Unf",
    "BsmtFinSF2": 0.0,
    "BsmtUnfSF": total_bsmt_sf * 0.4 if total_bsmt_sf > 0 else 0.0,
    "TotalBsmtSF": float(total_bsmt_sf),
    "Heating": "GasA",
    "HeatingQC": default_vals.get("HeatingQC", "Ex"),
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "1stFlrSF": first_flr_sf,
    "2ndFlrSF": second_flr_sf,
    "LowQualFinSF": 0,
    "GrLivArea": gr_liv_area,
    "BsmtFullBath": 1 if total_bsmt_sf > 500 else 0,
    "BsmtHalfBath": 0,
    "FullBath": full_bath,
    "HalfBath": int(default_vals.get("HalfBath", 0)),
    "BedroomAbvGr": bedrooms,
    "KitchenAbvGr": 1,
    "KitchenQual": kitchen_qual,
    "TotRmsAbvGrd": tot_rooms,
    "Functional": "Typ",
    "Fireplaces": fireplaces,
    "FireplaceQu": "Gd" if fireplaces > 0 else None,
    "GarageType": "Attchd" if garage_cars > 0 else None,
    "GarageYrBlt": float(year_built) if garage_cars > 0 else None,
    "GarageFinish": "RFn" if garage_cars > 0 else None,
    "GarageCars": int(garage_cars),
    "GarageArea": float(garage_area),
    "GarageQual": "TA" if garage_cars > 0 else None,
    "GarageCond": "TA" if garage_cars > 0 else None,
    "PavedDrive": "Y",
    "WoodDeckSF": 120.0,
    "OpenPorchSF": 40.0,
    "EnclosedPorch": 0.0,
    "3SsnPorch": 0.0,
    "ScreenPorch": 0.0,
    "PoolArea": 0.0,
    "PoolQC": None,
    "Fence": None,
    "MiscFeature": None,
    "MiscVal": 0.0,
    "MoSold": 6,
    "YrSold": 2010,
    "SaleType": "WD",
    "SaleCondition": "Normal",
}

# Run prediction
try:
    predicted_val = predict(house_data)
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns([2, 1, 1])
    with kpi_col1:
        st.metric(
            label="🏷️ Predicted Sale Price",
            value=f"${predicted_val:,.0f}",
            delta=f"Based on {neighborhood} market"
        )
    with kpi_col2:
        st.metric(label="Est. Price / Sq Ft", value=f"${predicted_val / max(1, gr_liv_area):,.0f} / sqft")
    with kpi_col3:
        st.metric(label="House Age at Sale", value=f"{2010 - year_built} years")

    # Expandable summary of submitted payload
    with st.expander("🔍 View Raw Features Processed by Pipeline"):
        st.json(house_data)

except Exception as err:
    st.error(f"Inference error: {err}")

# ── Visual Insights Section ───────────────────────────────────────────────────
st.subheader("📊 Model Artifacts & Explainability")
tab1, tab2, tab3 = st.tabs(["Global SHAP Feature Importance", "EDA Distributions", "Monitoring & Drift"])

with tab1:
    shap_path = PROJECT_ROOT / "data" / "processed" / "shap_summary.png"
    if shap_path.exists():
        st.image(str(shap_path), caption="SHAP Summary Plot — Top features driving predictions", use_container_width=True)
    else:
        st.info("SHAP plot not yet generated.")

with tab2:
    eda_dir = PROJECT_ROOT / "data" / "processed" / "eda"
    if eda_dir.exists():
        e_col1, e_col2 = st.columns(2)
        target_dist = eda_dir / "target_distribution.png"
        corr_map = eda_dir / "correlation_heatmap.png"
        if target_dist.exists():
            e_col1.image(str(target_dist), caption="Target Distribution", use_container_width=True)
        if corr_map.exists():
            e_col2.image(str(corr_map), caption="Correlation Heatmap", use_container_width=True)

with tab3:
    drift_json = PROJECT_ROOT / "monitoring" / "drift_summary.json"
    if drift_json.exists():
        import json
        with open(drift_json) as f:
            drift_data = json.load(f)
        st.write("**Evidently AI Drift Check Status:**")
        st.json(drift_data)
