# House Price Prediction — Task Tracker

## Phase 0 — Setup
- [x] Init repo, git, virtualenv
- [x] Install deps (requirements.txt)
- [x] Download Ames dataset via Kaggle API → data/raw/
- [x] Test MLflow UI local server

## Phase 1 — EDA
- [x] Check shape, dtypes, missing % per column
- [x] Plot SalePrice histogram (raw)
- [x] Plot log1p(SalePrice) histogram, confirm normalized
- [x] Scatter GrLivArea vs SalePrice, flag outlier indices
- [x] Scatter TotalBsmtSF, LotArea vs SalePrice, check outliers
- [x] Correlation heatmap, top 15 features vs target
- [x] Flag multicollinear pairs (GarageCars/GarageArea etc.)
- [x] Boxplot Neighborhood vs SalePrice
- [x] Check cardinality per categorical col
- [x] Document: outlier rows to drop
- [x] Document: missing-value semantics per column (meaningful-NaN vs true-missing)
- [x] Document: encoding strategy per categorical column

## Phase 2 — Feature Engineering
- [x] data_prep.py: load_data()
- [x] data_prep.py: remove_outliers()
- [x] data_prep.py: get_cv_folds() (5-fold)
- [x] Fill meaningful-NaN cols with "None"
- [x] Impute LotFrontage by Neighborhood median
- [x] Impute Electrical mode
- [x] Create TotalSF, HouseAge, RemodAge, TotalBath, HasPool, HasGarage
- [x] Ordinal-encode quality columns (manual map)
- [x] One-hot encode low-cardinality nominal cols
- [x] Target-encode Neighborhood (fit inside CV fold only)
- [x] Log-transform skewed numeric features (|skew| > 0.75)
- [x] StandardScaler for linear-model pipeline only
- [x] Build FeatureEngineer class (fit/transform)
- [x] Serialize fitted pipeline → feature_pipeline.pkl
- [x] Unit test: no NaNs after transform
- [x] Unit test: verify target-encoding leakage-free

## Phase 3 — Modeling
- [x] Baseline LinearRegression, sanity check pipeline
- [x] Ridge / Lasso / ElasticNet
- [x] RandomForestRegressor baseline
- [x] XGBRegressor
- [x] LGBMRegressor
- [x] 5-fold CV, refit FeatureEngineer per fold
- [x] Log all runs to MLflow (params + metrics)
- [x] Build model comparison table (mean/std RMSE, train time)
- [x] Select best model, document reasoning
- [x] (Optional) StackingRegressor, compare vs best single model

## Phase 4 — Hyperparameter Tuning
- [x] Define Optuna search space (XGBoost/LightGBM)
- [x] Run 50-100 trials
- [x] Inspect param importance (optuna.visualization)
- [x] Refit best-param model on full train data
- [x] Log final "production candidate" run to MLflow
- [x] Serialize final model → best_model.pkl

## Phase 5 — Explainability
- [x] SHAP TreeExplainer on final model
- [x] Generate + save SHAP summary plot (global importance)
- [x] Log SHAP plot to MLflow
- [x] Implement explain_single_prediction() (local/force plot)
- [x] Sanity check: top SHAP features match domain intuition

## Phase 6 — Deployment
- [x] Define Pydantic HouseFeatures schema
- [x] Define PredictionResponse schema
- [x] model_loader.py: load model + pipeline once at startup
- [x] FastAPI POST /predict endpoint
- [x] FastAPI GET /health endpoint
- [x] Write Dockerfile
- [x] Build + run Docker container locally
- [x] Test /predict via curl/Postman with sample payload
- [x] Test 422 response on malformed payload
- [x] (Optional) Streamlit frontend form → calls /predict

## Phase 7 — Monitoring
- [x] Save reference_data.csv (training distribution snapshot)
- [x] Create simulated drifted batch (perturbed copy)
- [x] Evidently AI DataDriftPreset report: reference vs new batch
- [x] Verify report flags perturbed features correctly
- [x] Document retrain-trigger condition based on drift result

## Final
- [x] Write README.md (setup, usage, architecture summary)
- [x] Clean requirements.txt (pin versions)
- [x] Write resume line / project summary for portfolio
- [x] Push to GitHub, verify .gitignore excludes data/models/mlruns
