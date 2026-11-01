"""
schema.py — Pydantic request/response models for the FastAPI prediction endpoint.

HouseFeatures covers the raw Ames Housing feature set (pre-engineering).
All optional fields default to typical/neutral values so partial inputs work.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    """
    Input schema: raw house property features.
    Mirrors the Kaggle Ames Housing train.csv columns (excluding Id and SalePrice).
    """

    # ── Lot / Location ──────────────────────────────────────────────────────────
    MSSubClass: int = Field(20, description="Building class (20=1-story 1946+)")
    MSZoning: str = Field("RL", description="General zoning classification")
    LotFrontage: Optional[float] = Field(None, description="Linear feet of street connected to property")
    LotArea: int = Field(9600, description="Lot size in square feet")
    Street: str = Field("Pave", description="Road access type")
    Alley: Optional[str] = Field(None, description="Alley access type")
    LotShape: str = Field("Reg", description="General shape of property")
    LandContour: str = Field("Lvl", description="Flatness of property")
    Utilities: str = Field("AllPub", description="Type of utilities available")
    LotConfig: str = Field("Inside", description="Lot configuration")
    LandSlope: str = Field("Gtl", description="Slope of property")
    Neighborhood: str = Field("NAmes", description="Physical locations within Ames city limits")
    Condition1: str = Field("Norm", description="Proximity to main road or railroad")
    Condition2: str = Field("Norm", description="Proximity to main road or railroad (if second)")

    # ── Building ────────────────────────────────────────────────────────────────
    BldgType: str = Field("1Fam", description="Type of dwelling")
    HouseStyle: str = Field("1Story", description="Style of dwelling")
    OverallQual: int = Field(5, ge=1, le=10, description="Overall material and finish quality")
    OverallCond: int = Field(5, ge=1, le=10, description="Overall condition rating")
    YearBuilt: int = Field(1975, description="Original construction date")
    YearRemodAdd: int = Field(1975, description="Remodel date")
    RoofStyle: str = Field("Gable", description="Type of roof")
    RoofMatl: str = Field("CompShg", description="Roof material")
    Exterior1st: str = Field("VinylSd", description="Exterior covering on house")
    Exterior2nd: str = Field("VinylSd", description="Exterior covering on house (if more than one material)")
    MasVnrType: str = Field("None", description="Masonry veneer type")
    MasVnrArea: float = Field(0.0, description="Masonry veneer area in square feet")
    ExterQual: str = Field("TA", description="Exterior material quality")
    ExterCond: str = Field("TA", description="Present condition of the material on exterior")
    Foundation: str = Field("PConc", description="Type of foundation")

    # ── Basement ────────────────────────────────────────────────────────────────
    BsmtQual: Optional[str] = Field(None, description="Height of the basement")
    BsmtCond: Optional[str] = Field(None, description="General condition of the basement")
    BsmtExposure: Optional[str] = Field(None, description="Walkout or garden level basement walls")
    BsmtFinType1: Optional[str] = Field(None, description="Quality of basement finished area")
    BsmtFinSF1: float = Field(0.0, description="Type 1 finished square feet")
    BsmtFinType2: Optional[str] = Field(None, description="Quality of second finished area")
    BsmtFinSF2: float = Field(0.0, description="Type 2 finished square feet")
    BsmtUnfSF: float = Field(0.0, description="Unfinished square feet of basement area")
    TotalBsmtSF: float = Field(0.0, description="Total square feet of basement area")
    BsmtFullBath: int = Field(0, description="Basement full bathrooms")
    BsmtHalfBath: int = Field(0, description="Basement half bathrooms")

    # ── Utilities ───────────────────────────────────────────────────────────────
    Heating: str = Field("GasA", description="Type of heating")
    HeatingQC: str = Field("TA", description="Heating quality and condition")
    CentralAir: str = Field("Y", description="Central air conditioning")
    Electrical: str = Field("SBrkr", description="Electrical system")

    # ── Living area ─────────────────────────────────────────────────────────────
    FirstFlrSF: int = Field(1000, validation_alias="1stFlrSF")
    SecondFlrSF: int = Field(0, validation_alias="2ndFlrSF")
    LowQualFinSF: int = Field(0, description="Low quality finished square feet (all floors)")
    GrLivArea: int = Field(1500, description="Above grade (ground) living area square feet")
    FullBath: int = Field(2, description="Full bathrooms above grade")
    HalfBath: int = Field(0, description="Half baths above grade")
    BedroomAbvGr: int = Field(3, description="Number of bedrooms above basement level")
    KitchenAbvGr: int = Field(1, description="Number of kitchens above grade")
    KitchenQual: str = Field("TA", description="Kitchen quality")
    TotRmsAbvGrd: int = Field(6, description="Total rooms above grade (not including bathrooms)")
    Functional: str = Field("Typ", description="Home functionality rating")
    Fireplaces: int = Field(0, description="Number of fireplaces")
    FireplaceQu: Optional[str] = Field(None, description="Fireplace quality")

    # ── Garage ──────────────────────────────────────────────────────────────────
    GarageType: Optional[str] = Field(None, description="Garage location")
    GarageYrBlt: Optional[float] = Field(None, description="Year garage was built")
    GarageFinish: Optional[str] = Field(None, description="Interior finish of garage")
    GarageCars: int = Field(2, description="Size of garage in car capacity")
    GarageArea: float = Field(500.0, description="Size of garage in square feet")
    GarageQual: Optional[str] = Field(None, description="Garage quality")
    GarageCond: Optional[str] = Field(None, description="Garage condition")
    PavedDrive: str = Field("Y", description="Paved driveway")

    # ── Outdoor ─────────────────────────────────────────────────────────────────
    WoodDeckSF: float = Field(0.0, description="Wood deck area in square feet")
    OpenPorchSF: float = Field(0.0, description="Open porch area in square feet")
    EnclosedPorch: float = Field(0.0, description="Enclosed porch area in square feet")
    ThreeSsnPorch: float = Field(0.0, validation_alias="3SsnPorch")
    ScreenPorch: float = Field(0.0, description="Screen porch area in square feet")
    PoolArea: float = Field(0.0, description="Pool area in square feet")
    PoolQC: Optional[str] = Field(None, description="Pool quality")
    Fence: Optional[str] = Field(None, description="Fence quality")
    MiscFeature: Optional[str] = Field(None, description="Miscellaneous feature not covered in other categories")
    MiscVal: float = Field(0.0, description="Dollar value of miscellaneous feature")

    # ── Sale ────────────────────────────────────────────────────────────────────
    MoSold: int = Field(6, ge=1, le=12, description="Month sold")
    YrSold: int = Field(2010, description="Year sold")
    SaleType: str = Field("WD", description="Type of sale")
    SaleCondition: str = Field("Normal", description="Condition of sale")

    model_config = {"populate_by_name": True}

    def to_dict(self) -> dict:
        """Convert to dict using field names matching original CSV column names."""
        d = self.model_dump(by_alias=False)
        # Remap internal field names back to original CSV names
        rename = {"FirstFlrSF": "1stFlrSF", "SecondFlrSF": "2ndFlrSF", "ThreeSsnPorch": "3SsnPorch"}
        return {rename.get(k, k): v for k, v in d.items()}


class PredictionResponse(BaseModel):
    """Output schema returned by POST /predict."""
    predicted_price: float = Field(..., description="Predicted sale price in USD")
    currency: str = Field("USD", description="Currency of prediction")
    model_version: str = Field("1.0", description="Model version identifier")


class BatchPredictionRequest(BaseModel):
    """Input for POST /predict/batch."""
    houses: list[HouseFeatures]


class BatchPredictionResponse(BaseModel):
    """Output for POST /predict/batch."""
    predictions: list[float]
    currency: str = "USD"
    count: int


class HealthResponse(BaseModel):
    """Output for GET /health."""
    status: str
    model_loaded: bool
    pipeline_loaded: bool
    model_version: str = "1.0"
