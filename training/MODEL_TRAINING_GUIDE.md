# Model Training Guide

This guide walks you through training all models for the mall movement tracking project.

## Prerequisites

Before training models, ensure feature engineering is complete:

```bash
# Verify feature engineering
python features/verify_feature_engineering.py
```

This will check:
- ✅ Processed data exists
- ✅ Feature engineering module is available
- ✅ Engineered features are created
- ✅ All required features are present

---

## Training Workflow

### Step 1: Verify Feature Engineering

```bash
python features/verify_feature_engineering.py
```

**Expected Output:**
```
✓ ALL CHECKS PASSED!
✓ Feature engineering is complete and ready for model training!
```

If checks fail, run feature engineering first:
```bash
python features/run_feature_engineering.py
```

---

### Step 2: Train Classification Models

Classification models predict the next zone a customer will visit.

```bash
python training/train_classification.py
```

**What it does:**
- Loads processed data
- Applies feature engineering
- Trains 3 models:
  - **Random Forest** - Best for complex patterns
  - **Decision Tree** - Baseline model
  - **XGBoost** - Gradient boosting
- Saves models to `models/classification/`
- Saves metrics to `results/classification/metrics.json`

**Expected Output:**
```
Training Classification Models
==================================================
Loading processed data...
Engineering features...
Training set size: (X, Y)
Test set size: (X, Y)

Training Random Forest...
Random Forest Accuracy: 0.XXXX
Random Forest ROC-AUC: 0.XXXX

Training Decision Tree...
Decision Tree Accuracy: 0.XXXX

Training XGBoost...
XGBoost Accuracy: 0.XXXX
XGBoost ROC-AUC: 0.XXXX

Training Complete!
Best model: random_forest
```

**Output Files:**
- `models/classification/zone_rf.pkl` - Random Forest model
- `models/classification/baseline_dt.pkl` - Decision Tree model
- `models/classification/zone_xgb.pkl` - XGBoost model
- `results/classification/metrics.json` - Performance metrics

---

### Step 3: Train Clustering Models

Clustering models group customers with similar movement patterns.

```bash
python training/train_clustering.py
```

**What it does:**
- Loads processed data
- Applies feature engineering
- Scales features
- Trains 2 models:
  - **K-Means** - Groups customers into k clusters
  - **DBSCAN** - Density-based clustering
- Saves models to `models/clustering/`
- Saves metrics to `results/clustering/silhouette_score.json`

**Expected Output:**
```
Training Clustering Models
==================================================
Loading processed data...
Engineering features...
Data shape: (X, Y)

Training K-Means with 5 clusters...
K-Means Silhouette Score: 0.XXXX

Training DBSCAN...
DBSCAN found X clusters
DBSCAN Silhouette Score: 0.XXXX

Training Complete!
```

**Output Files:**
- `models/clustering/kmeans.pkl` - K-Means model
- `models/clustering/dbscan.pkl` - DBSCAN model
- `models/preprocessing/scaler.pkl` - Feature scaler
- `results/clustering/silhouette_score.json` - Clustering metrics

---

### Step 4: Train Forecasting Models

Forecasting models predict future traffic patterns.

```bash
python training/train_forecasting.py
```

**What it does:**
- Loads processed data
- Detects datetime and value columns
- Creates time series
- Trains 2 models:
  - **ARIMA** - Statistical time series model
  - **Prophet** - Facebook's forecasting tool
- Saves models to `models/forecasting/`
- Saves metrics to `results/forecasting/rmse.json`

**Expected Output:**
```
Training Forecasting Models
==================================================
Loading processed data...
Time series length: XXXX
Date range: YYYY-MM-DD to YYYY-MM-DD

Training ARIMA...
ARIMA RMSE: XXXX.XX
ARIMA MAE: XXXX.XX

Training Prophet...
Prophet RMSE: XXXX.XX
Prophet MAE: XXXX.XX

Training Complete!
```

**Output Files:**
- `models/forecasting/arima.pkl` - ARIMA model
- `models/forecasting/prophet_model.pkl` - Prophet model
- `results/forecasting/rmse.json` - Forecasting metrics

---

## Training All Models at Once

You can train all models sequentially:

```bash
# Windows PowerShell
python features/verify_feature_engineering.py
python training/train_classification.py
python training/train_clustering.py
python training/train_forecasting.py

# Or create a batch script (train_all.bat)
@echo off
echo Verifying feature engineering...
python features/verify_feature_engineering.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo Training classification models...
python training/train_classification.py

echo Training clustering models...
python training/train_clustering.py

echo Training forecasting models...
python training/train_forecasting.py

echo All models trained successfully!
```

---

## Model Training Details

### Classification Models

**Purpose:** Predict next zone visit

**Features Used:**
- Temporal features (hour, day, month, etc.)
- User activity features
- Zone popularity features
- Movement pattern features

**Target Variable:**
- Auto-detected zone/location column

**Evaluation Metrics:**
- Accuracy
- ROC-AUC (for binary/multi-class)

---

### Clustering Models

**Purpose:** Group similar customer behaviors

**Features Used:**
- All numeric engineered features
- Scaled using StandardScaler

**Parameters:**
- K-Means: n_clusters=5 (adjustable)
- DBSCAN: eps=0.5, min_samples=5

**Evaluation Metrics:**
- Silhouette Score
- Number of clusters
- Number of noise points (DBSCAN)

---

### Forecasting Models

**Purpose:** Predict future traffic patterns

**Data Requirements:**
- Datetime column (auto-detected)
- Numeric value column (auto-detected)

**Models:**
- ARIMA: (1,1,1) order
- Prophet: Default parameters

**Evaluation Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

---

## Troubleshooting

### Issue: "No numeric feature columns found"
**Solution:** Run feature engineering first
```bash
python features/run_feature_engineering.py
```

### Issue: "Target column not found"
**Solution:** Check your data has zone/location columns, or modify training script to specify target column

### Issue: "Memory error"
**Solution:** 
- Reduce dataset size for testing
- Use smaller models
- Process data in chunks

### Issue: "Module not found"
**Solution:** Install missing dependencies
```bash
pip install -r requirements.txt
```

### Issue: "ARIMA/Prophet training failed"
**Solution:**
- Check datetime column is properly formatted
- Ensure sufficient data points (minimum 50+)
- Check for missing values in time series

---

## Model Performance

After training, check results:

```bash
# Classification results
cat results/classification/metrics.json

# Clustering results
cat results/clustering/silhouette_score.json

# Forecasting results
cat results/forecasting/rmse.json
```

---

## Next Steps

After training all models:

1. **View Results:**
   - Check metrics in `results/` folders
   - Review model performance

2. **Use Models:**
   - Streamlit Dashboard: `streamlit run streamlit_app/app.py`
   - API: `cd api && uvicorn app:app --reload`

3. **Improve Models:**
   - Tune hyperparameters: `python training/hyperparameter_tuning.py`
   - Run experiments: `python training/experiment_runner.py`

---

## Quick Reference

```bash
# 1. Verify feature engineering
python features/verify_feature_engineering.py

# 2. Train all models
python training/train_classification.py
python training/train_clustering.py
python training/train_forecasting.py

# 3. Check results
ls results/classification/
ls results/clustering/
ls results/forecasting/
```

---

## Support

For issues or questions:
- Check `WORKFLOW.md` for overall workflow
- Review `features/README.md` for feature engineering details
- Check model training scripts for specific parameters

