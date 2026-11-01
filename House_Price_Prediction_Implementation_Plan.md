# House Price Prediction — Full Implementation Plan
**End-to-End ML System: EDA → Model → Deployment → Monitoring**
Author: Vedansh | Version 1.0

---

## 1. Project Overview

### 1.1 Vision
Full production-style ML system predicting house sale price from property features. Not a notebook-only project — includes feature engineering, experiment tracking, deployment as API, and drift monitoring. Demonstrates production ML thinking, not just modeling skill.

### 1.2 Goals
- Clean, leakage-free feature engineering pipeline
- Multiple model comparison with rigorous CV-based evaluation
- Explainable predictions (SHAP)
- Deployed, callable API (FastAPI + Docker)
- Basic production monitoring (data drift detection)

### 1.3 Non-Goals (v1)
- No real-time streaming data ingestion
- No live retraining pipeline (manual retrain only)
- No multi-region/cloud deployment (local Docker sufficient for portfolio)

### 1.4 Success Criteria
- CV RMSE (log-scale) competitive with public Kaggle leaderboard benchmarks (~0.11-0.13 range for Ames)
- API returns valid prediction for arbitrary valid input in <200ms
- SHAP explanation generated for any single prediction
- Drift report runs successfully on a simulated "new data" batch

---

## 2. Dataset

**Primary**: Kaggle "House Prices - Advanced Regression Techniques" (Ames Housing), ~1460 train rows, 79 features, target `SalePrice`.

**Acquisition**: Download via Kaggle API (`kaggle competitions download -c house-prices-advanced-regression-techniques`) or manual download to `data/raw/`.

---

## 3. Repository Structure (Molecular Detail)

```
house-price-ml/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/
│       ├── train_processed.csv
│       └── feature_pipeline.pkl        # fitted encoders/scalers, saved for inference reuse
│
├── notebooks/
│   └── eda.ipynb                        # exploration only, no production logic lives here
│
├── src/
│   ├── __init__.py
│   ├── config.py                        # paths, constants, feature lists (numeric/categorical/ordinal)
│   ├── data_prep.py                     # load_data(), train_test_split_cv(), outlier_removal()
│   ├── feature_eng.py                   # FeatureEngineer class: fit_transform(), transform()
│   ├── train.py                         # train_models(), cross_validate(), save_best_model()
│   ├── tune.py                          # Optuna study definitions per model
│   ├── evaluate.py                      # compute_metrics(), residual_analysis()
│   ├── explain.py                       # generate_shap_summary(), explain_single_prediction()
│   └── predict.py                       # load_model(), predict(input_dict) -> price
│
├── models/
│   └── best_model.pkl                   # serialized final model (joblib)
│
├── api/
│   ├── main.py                           # FastAPI app, /predict, /health endpoints
│   ├── schema.py                         # Pydantic HouseFeatures input model, PredictionResponse
│   └── model_loader.py                   # loads model + feature_pipeline once at startup
│
├── monitoring/
│   ├── drift_report.py                   # Evidently AI report generation
│   └── reference_data.csv                # snapshot of training distribution for comparison
│
├── mlruns/                                # MLflow tracking store (local)
├── tests/
│   ├── test_feature_eng.py                # verify no leakage, correct transforms
│   ├── test_predict.py                    # verify predict() returns valid float for valid input
│   └── test_api.py                        # hit /predict endpoint, check response schema
│
├── Dockerfile
├── requirements.txt
├── .gitignore                              # data/, mlruns/, models/*.pkl (large files)
└── README.md
```

---

## 4. Phase-by-Phase Plan (Molecular Detail)

### PHASE 0 — Environment Setup (0.5 day)
1. Init repo, git, virtualenv
2. Install deps: `pandas numpy scikit-learn xgboost lightgbm shap mlflow fastapi uvicorn pydantic evidently optuna joblib matplotlib seaborn`
3. Download dataset via Kaggle API, place in `data/raw/`
4. Init MLflow tracking: `mlflow ui` local server test

**Exit criteria**: Data loads in pandas, MLflow UI accessible locally.

---

### PHASE 1 — EDA (1-2 days)

**1.1 Basic checks (`notebooks/eda.ipynb`)**
- `df.shape`, `df.dtypes`, `df.isnull().sum()` sorted descending
- Identify columns >40% missing (candidates for drop vs meaningful-NaN encoding)

**1.2 Target analysis**
- Plot `SalePrice` histogram — confirm right-skew
- Plot `np.log1p(SalePrice)` histogram — confirm near-normal after transform
- Decision: model on log-target throughout, inverse-transform (`np.expm1`) only at final prediction output

**1.3 Outlier detection**
- Scatter `GrLivArea` vs `SalePrice` — Ames dataset docs flag known outliers (very large GrLivArea, abnormally low price) — identify and note indices for removal in Phase 2
- Repeat check for `TotalBsmtSF`, `LotArea` vs target

**1.4 Correlation analysis**
- Numeric feature correlation matrix (heatmap) vs `SalePrice`
- Flag top 15 correlated features — cross-check against domain intuition (OverallQual, GrLivArea, GarageCars, TotalBsmtSF expected high)
- Flag multicollinear pairs (e.g. GarageCars/GarageArea, TotalBsmtSF/1stFlrSF) — note for later (tree models handle fine, linear models need care)

**1.5 Categorical analysis**
- Boxplot `Neighborhood` vs `SalePrice` — high variance, likely strong predictor
- Check cardinality of each categorical column (`nunique()`) — decide encoding strategy per column in Phase 2

**Exit criteria**: Documented list of (a) outlier row indices to remove, (b) missing-value semantics per column (meaningful-NaN vs true-missing), (c) encoding strategy assignment per categorical column.

---

### PHASE 2 — Feature Engineering (2-3 days)

**2.1 Data prep (`src/data_prep.py`)**
- `load_data()`: read train/test CSVs
- `remove_outliers(df)`: drop flagged indices from Phase 1
- `get_cv_folds(df, k=5)`: `KFold` or `StratifiedKFold` on price-bins for balanced fold distribution

**2.2 Missing value handling (`src/feature_eng.py`)**
- Meaningful-NaN columns (`PoolQC`, `Fence`, `Alley`, `FireplaceQu`, `GarageType` etc.) → fill with explicit `"None"` category string
- True-missing numeric (`LotFrontage`) → impute with **median grouped by Neighborhood** (more accurate than global median — captures local lot-size norms)
- True-missing categorical (`Electrical` — 1 row) → mode impute

**2.3 Feature creation**
```python
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
df['HasPool'] = (df['PoolArea'] > 0).astype(int)
df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
```

**2.4 Encoding strategy**
- Ordinal quality columns (`ExterQual`, `KitchenQual`, `BsmtQual` etc., values Ex/Gd/TA/Fa/Po/None) → manual ordinal map `{'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}`
- Low-cardinality nominal (`MSZoning`, `LotShape` etc.) → one-hot encoding
- High-cardinality nominal (`Neighborhood`, 25 unique) → target encoding (mean target per category), **fit only on train fold within CV loop** to prevent leakage — implement as sklearn-compatible transformer fit inside each CV fold, not globally beforehand

**2.5 Skew correction**
- `scipy.stats.skew()` on all numeric features, log-transform (`np.log1p`) any with |skew| > 0.75

**2.6 Scaling**
- `StandardScaler` fit on train fold, applied to linear model inputs only
- Tree models (XGBoost/LightGBM/RF) use unscaled features — skip scaling step for those pipelines

**2.7 Build `FeatureEngineer` class**
- `fit(df)`: learn imputation values, target-encoding maps, scaler params — from train fold only
- `transform(df)`: apply learned transforms to any new data (train fold, val fold, or true test/inference input)
- Serialize fitted instance via `joblib.dump()` to `data/processed/feature_pipeline.pkl` — **this exact object gets reused at inference time in the API**, critical for train/serve consistency

**Testing (`tests/test_feature_eng.py`)**
- Verify `transform()` on held-out fold produces no NaNs
- Verify target-encoding fitted only on train fold data (assert val-fold rows didn't influence encoding map)

**Exit criteria**: `FeatureEngineer` produces clean, leakage-free feature matrix, unit tests pass.

---

### PHASE 3 — Modeling (2-3 days)

**3.1 Baseline (`src/train.py`)**
- Plain `LinearRegression` on processed features — sanity baseline, expect moderate RMSE, use to confirm pipeline correctness before complex models

**3.2 Regularized linear models**
- `Ridge`, `Lasso`, `ElasticNet` — default alpha first, note Lasso's zeroed coefficients as informal feature-importance check

**3.3 Tree ensembles**
- `RandomForestRegressor` baseline
- `XGBRegressor` — objective `reg:squarederror`, eval on log-target
- `LGBMRegressor` — typically fastest, often competitive/better RMSE

**3.4 Cross-validation protocol**
- 5-fold CV, **fit `FeatureEngineer` fresh inside each fold** (not once globally) — reinforces leakage-free discipline
- Metric: RMSE on log-target (matches Kaggle standard); also compute back-transformed MAE in dollars for interpretability
- Log every model's per-fold and mean CV RMSE to MLflow (`mlflow.log_metric`, `mlflow.log_param` for model type)

**3.5 Model comparison table**
- Record: model name, mean CV RMSE, std CV RMSE, training time
- Select best single model by lowest mean RMSE (tie-break: prefer simpler/faster model if within noise)

**3.6 Stacking (stretch, optional)**
- `StackingRegressor` (sklearn) combining top 2-3 models with a meta Ridge learner
- Compare stacked CV RMSE vs best single model — include only if improvement is real, not noise (check against std dev)

**Exit criteria**: Model comparison table complete, best model selected with documented reasoning, all runs logged in MLflow.

---

### PHASE 4 — Hyperparameter Tuning (1-2 days)

**4.1 Tuning setup (`src/tune.py`)**
- Use **Optuna** for Bayesian search (more sample-efficient than GridSearch for tree models with many params)
- XGBoost/LightGBM search space: `max_depth` [3-10], `learning_rate` [0.01-0.3, log scale], `n_estimators` [100-1000], `subsample` [0.6-1.0], `colsample_bytree` [0.6-1.0], `reg_alpha`/`reg_lambda` [0-10]
- Objective function: return mean CV RMSE (log-target) for given param set
- Run 50-100 trials, use `optuna.visualization` to inspect param importance

**4.2 Final refit**
- Refit best-param model on full training data (no held-out fold needed now — tuning already validated via CV)
- Log final params + final CV metric to MLflow as the "production candidate" run

**4.3 Serialize**
- `joblib.dump(final_model, 'models/best_model.pkl')`

**Exit criteria**: Tuned model outperforms (or matches within noise) untuned baseline from Phase 3, tracked in MLflow.

---

### PHASE 5 — Explainability (1 day)

**5.1 Global explanation (`src/explain.py`)**
- `shap.TreeExplainer(model)` (fast, exact for tree models)
- Generate SHAP summary plot (global feature importance + direction of effect) on validation set
- Save plot as PNG artifact, log to MLflow

**5.2 Local explanation**
- `explain_single_prediction(input_row)`: SHAP force plot for one house — shows which features pushed price up/down from baseline
- Wrap as reusable function — this becomes an optional API endpoint later (`/explain`)

**Exit criteria**: Summary plot generated and interpretable; matches domain intuition (OverallQual, GrLivArea, TotalSF should rank high).

---

### PHASE 6 — Deployment (2-3 days)

**6.1 Pydantic schema (`api/schema.py`)**
- `HouseFeatures` model: field per raw input feature (pre-engineering), correct types (int/float/str), sensible defaults where reasonable (e.g. `PoolQC: Optional[str] = None`)
- `PredictionResponse`: `{predicted_price: float, currency: str = "USD"}`

**6.2 Model loader (`api/model_loader.py`)**
- Load `best_model.pkl` and `feature_pipeline.pkl` once at app startup (module-level, not per-request — avoid reload overhead)

**6.3 FastAPI app (`api/main.py`)**
- `POST /predict`: accept `HouseFeatures` JSON → convert to DataFrame row → `feature_pipeline.transform()` → `model.predict()` → `np.expm1()` inverse log-transform → return `PredictionResponse`
- `GET /health`: simple liveness check, returns model version/load status
- Input validation errors (Pydantic) auto-return 422 with clear message

**6.4 Docker**
- `Dockerfile`: base `python:3.11-slim`, copy `src/`, `api/`, `models/`, install requirements, `CMD uvicorn api.main:app --host 0.0.0.0 --port 8000`
- Build + run locally, test via `curl`/Postman with sample house JSON

**6.5 Optional Streamlit frontend**
- Simple form (sliders/dropdowns for key features) → calls local FastAPI `/predict` → displays result
- Good live-demo artifact for interviews

**Testing (`tests/test_api.py`)**
- Send valid payload → assert 200 + float response
- Send malformed payload (missing required field) → assert 422

**Exit criteria**: API runs in Docker container, returns correct predictions for known validation-set rows (cross-check against offline model output — no drift from serialization).

---

### PHASE 7 — Monitoring (1-2 days)

**7.1 Reference snapshot**
- Save a snapshot of training feature distributions to `monitoring/reference_data.csv` — this is the "expected" baseline distribution

**7.2 Simulated drift batch**
- Create a synthetic "new data" batch: either hold out a time-based slice of original data (if timestamps available) or artificially perturb a copy (e.g. shift `YearBuilt` distribution, inflate `LotArea` values) to simulate real-world drift

**7.3 Drift report (`monitoring/drift_report.py`)**
- Use **Evidently AI**: `Report(metrics=[DataDriftPreset()])`, compare reference vs "new" batch
- Generate HTML report — flags which features drifted significantly (statistical tests per column: KS-test for numeric, chi-square for categorical)
- Document: if drift detected on top-importance features (per Phase 5 SHAP ranking) → flag as retrain-trigger condition

**Exit criteria**: Drift report generates successfully, correctly flags the artificially-perturbed features as drifted (sanity check that monitoring actually works).

---

## 5. Dependencies (requirements.txt)

```
pandas
numpy
scikit-learn
xgboost
lightgbm
shap
mlflow
optuna
fastapi
uvicorn
pydantic
evidently
joblib
matplotlib
seaborn
streamlit
```

---

## 6. Timeline Summary

| Phase | Duration | Cumulative |
|---|---|---|
| 0 — Setup | 0.5 day | 0.5 day |
| 1 — EDA | 1-2 days | ~2.5 days |
| 2 — Feature Engineering | 2-3 days | ~5.5 days |
| 3 — Modeling | 2-3 days | ~8.5 days |
| 4 — Hyperparameter Tuning | 1-2 days | ~10.5 days |
| 5 — Explainability | 1 day | ~11.5 days |
| 6 — Deployment | 2-3 days | ~14.5 days |
| 7 — Monitoring | 1-2 days | ~16.5 days |
| **Total** | | **~12-17 days solo** |

---

## 7. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Data leakage via global fit before CV split | High (invalidates all metrics) | Fit `FeatureEngineer`/target-encoding fresh inside each CV fold, never on full dataset upfront |
| Target-encoding overfits high-cardinality columns | Medium | Use smoothing/regularization in target encoder, or fall back to one-hot if overfit observed |
| Train/serve skew (API predictions differ from offline eval) | High | Reuse exact same serialized `FeatureEngineer` object at inference, never reimplement transform logic separately in API |
| Single train/test split misrepresents performance | Medium | Always report 5-fold CV mean ± std, never a single split number |
| Drift report never triggers real signal (synthetic-only testing) | Low | Explicitly test with deliberately perturbed batch to prove monitoring pipeline works end-to-end |

---

## 8. Resume Line (draft)

"Built end-to-end house price prediction system (Ames Housing dataset) — leakage-free feature engineering, XGBoost/LightGBM with Optuna tuning, MLflow experiment tracking, SHAP explainability, FastAPI+Docker deployment, and Evidently-based drift monitoring."
