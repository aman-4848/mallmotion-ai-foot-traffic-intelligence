# ML Training Workflow - Complete Guide

## Overview

This document explains how Machine Learning training works in the Mall Movement Tracking project, how all folders and files work together, and the complete workflow from feature engineering to trained models.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Folder Structure & Responsibilities](#folder-structure--responsibilities)
3. [Complete Workflow](#complete-workflow)
4. [How Components Work Together](#how-components-work-together)
5. [Training Process Details](#training-process-details)
6. [Model Types & Algorithms](#model-types--algorithms)
7. [File Dependencies](#file-dependencies)
8. [Data Flow Diagram](#data-flow-diagram)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
│  data/processed/merged data set.csv                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING LAYER                      │
│  features/feature_engineering.py                           │
│  → Missing values, Encoding, Temporal, Domain features      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              TRAINING LAYER                                 │
│  training/train_*.py                                        │
│  → Classification, Clustering, Forecasting                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              MODEL STORAGE LAYER                            │
│  models/classification/, clustering/, forecasting/          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              RESULTS LAYER                                  │
│  results/classification/, clustering/, forecasting/          │
└─────────────────────────────────────────────────────────────┘
```

---

## Folder Structure & Responsibilities

### 1. `data/` - Data Storage

**Purpose:** Stores all data files

```
data/
├── processed/
│   ├── merged data set.csv          # Original processed data
│   └── engineered_features.csv      # After feature engineering
└── sample/                          # Sample data files
```

**Responsibilities:**
- Store raw processed data
- Store engineered features
- Provide data access for training

**Key Files:**
- `merged data set.csv`: Input data (15,839 rows × 80 columns)
- `engineered_features.csv`: Output after feature engineering (15,839 rows × 110 columns)

---

### 2. `features/` - Feature Engineering

**Purpose:** Transform raw data into ML-ready features

```
features/
├── feature_engineering.py           # Main FeatureEngineer class
├── feature_config.yaml              # Configuration
├── run_feature_engineering.py      # Standalone execution
├── feature_analysis.py              # Analysis & visualization
└── verify_feature_engineering.py    # Verification script
```

**Responsibilities:**
- Handle missing values
- Extract temporal features
- Encode categorical variables
- Detect outliers
- Create domain-specific features
- Combine columns

**Key Functions:**
- `FeatureEngineer.engineer_features()`: Main pipeline
- `FeatureEngineer.handle_missing_values()`: Missing value imputation
- `FeatureEngineer.extract_datetime_features()`: Temporal features
- `FeatureEngineer.create_domain_features()`: Business logic features

---

### 3. `training/` - Model Training Scripts

**Purpose:** Train ML models using engineered features

```
training/
├── train_classification.py          # Classification models
├── train_clustering.py               # Clustering models
├── train_forecasting.py              # Forecasting models
├── hyperparameter_tuning.py          # Hyperparameter optimization
├── experiment_runner.py              # Experiment management
└── MODEL_TRAINING_GUIDE.md          # Training guide
```

**Responsibilities:**
- Load and prepare data
- Apply feature engineering
- Train multiple models
- Evaluate model performance
- Save trained models
- Save evaluation metrics

**Key Functions:**
- `prepare_data()`: Load data and apply feature engineering
- `train_*()`: Train specific model types
- `main()`: Orchestrate training process

---

### 4. `models/` - Trained Model Storage

**Purpose:** Store trained models and preprocessing objects

```
models/
├── classification/
│   ├── zone_rf.pkl                  # Random Forest model
│   ├── baseline_dt.pkl              # Decision Tree model
│   └── zone_xgb.pkl                 # XGBoost model
├── clustering/
│   ├── kmeans.pkl                   # K-Means model
│   └── dbscan.pkl                   # DBSCAN model
├── forecasting/
│   ├── arima.pkl                    # ARIMA model
│   └── prophet_model.pkl            # Prophet model
├── preprocessing/
│   ├── encoder.pkl                  # Label encoder
│   └── scaler.pkl                   # Feature scaler
├── load_model.py                    # Model loading utility
└── model_registry.json              # Model metadata
```

**Responsibilities:**
- Store trained models (`.pkl` files)
- Store preprocessing transformers
- Provide model loading utilities
- Track model metadata

**Model Files:**
- **Classification**: 3 models (RF, DT, XGBoost)
- **Clustering**: 2 models (K-Means, DBSCAN)
- **Forecasting**: 2 models (ARIMA, Prophet)
- **Preprocessing**: 2 transformers (Encoder, Scaler)

---

### 5. `results/` - Training Results

**Purpose:** Store model evaluation metrics and visualizations

```
results/
├── classification/
│   ├── metrics.json                 # Performance metrics
│   ├── confusion_matrix.png         # Confusion matrix plot
│   ├── roc_auc.png                  # ROC curve
│   └── feature_importance.png       # Feature importance
├── clustering/
│   ├── silhouette_score.json        # Clustering metrics
│   └── cluster_plot.png             # Cluster visualization
├── forecasting/
│   ├── rmse.json                    # Forecasting metrics
│   └── forecast_plot.png            # Forecast visualization
└── comparisons/
    ├── model_comparison_table.csv   # Model comparison
    └── best_model.txt               # Best model info
```

**Responsibilities:**
- Store evaluation metrics
- Store visualization plots
- Enable model comparison
- Track training history

---

### 6. `streamlit_app/utils/` - Utilities

**Purpose:** Shared utilities for data loading and model loading

```
streamlit_app/utils/
├── data_loader.py                   # Load processed data
├── model_loader.py                   # Load trained models
├── preprocess.py                     # Preprocessing utilities
└── charts.py                         # Visualization utilities
```

**Key Functions:**
- `load_processed_data()`: Load data from `data/processed/`
- `load_model()`: Load models from `models/`

---

## Complete Workflow

### Step-by-Step Process

```
1. DATA PREPARATION
   └── data/processed/merged data set.csv
       ↓
2. FEATURE ENGINEERING
   └── features/feature_engineering.py
       ├── Handle missing values
       ├── Extract datetime features
       ├── Encode categorical variables
       ├── Handle outliers
       ├── Create bins
       ├── Domain features
       └── Combine columns
       ↓
   └── data/processed/engineered_features.csv
       ↓
3. MODEL TRAINING
   └── training/train_*.py
       ├── Load engineered data
       ├── Split train/test
       ├── Train models
       ├── Evaluate performance
       └── Save models & metrics
       ↓
   └── models/*/*.pkl (trained models)
   └── results/*/metrics.json (evaluation results)
       ↓
4. MODEL USAGE
   └── streamlit_app/ (Dashboard)
   └── api/ (API endpoints)
```

---

## How Components Work Together

### 1. Data Loading → Feature Engineering

**Connection:**
```python
# In training scripts
from streamlit_app.utils.data_loader import load_processed_data
from features.feature_engineering import FeatureEngineer

# Load data
df = load_processed_data()  # From data/processed/merged data set.csv

# Engineer features
fe = FeatureEngineer()
df_engineered = fe.engineer_features(df)  # Uses features/feature_engineering.py
```

**Flow:**
- `data_loader.py` → Reads CSV from `data/processed/`
- `FeatureEngineer` → Transforms data using `features/feature_engineering.py`
- Output → Ready for ML training

---

### 2. Feature Engineering → Model Training

**Connection:**
```python
# In train_classification.py
def prepare_data():
    df = load_processed_data()
    fe = FeatureEngineer()
    df = fe.engineer_features(df)  # Apply feature engineering
    
    # Prepare for training
    X = df[feature_cols]
    y = df[target_col]
    return X, y
```

**Flow:**
- Training scripts import `FeatureEngineer`
- Apply feature engineering automatically
- Use engineered features for training

---

### 3. Model Training → Model Storage

**Connection:**
```python
# In train_classification.py
import joblib

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, MODEL_DIR / "zone_rf.pkl")  # Saves to models/classification/
```

**Flow:**
- Models trained in `training/` scripts
- Saved to `models/classification/`, `models/clustering/`, `models/forecasting/`
- Preprocessing objects saved to `models/preprocessing/`

---

### 4. Model Storage → Results

**Connection:**
```python
# In train_classification.py
results = {
    'random_forest': {'accuracy': 0.85, 'roc_auc': 0.92}
}

# Save results
with open(RESULTS_DIR / "metrics.json", 'w') as f:  # Saves to results/classification/
    json.dump(results, f)
```

**Flow:**
- Metrics calculated during training
- Saved to `results/` folders
- Visualizations generated and saved

---

### 5. Models → Application Usage

**Connection:**
```python
# In streamlit_app/utils/model_loader.py or api/app.py
import joblib

# Load model
model = joblib.load('models/classification/zone_rf.pkl')

# Make predictions
predictions = model.predict(X_new)
```

**Flow:**
- Models loaded from `models/` folders
- Used in Streamlit dashboard and API
- Predictions made on new data

---

## Training Process Details

### Classification Training (`train_classification.py`)

**Process:**
1. **Load Data**
   ```python
   df = load_processed_data()  # From data/processed/merged data set.csv
   ```

2. **Feature Engineering**
   ```python
   fe = FeatureEngineer()
   df = fe.engineer_features(df)  # 80 → 110 columns
   ```

3. **Prepare Features & Target**
   ```python
   # Auto-detect target (zone/location column)
   target_col = detect_target_column(df)
   
   # Select numeric features
   feature_cols = select_numeric_features(df)
   
   X = df[feature_cols]
   y = df[target_col]
   ```

4. **Split Data**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   ```

5. **Train Models**
   - Random Forest: `RandomForestClassifier(n_estimators=100)`
   - Decision Tree: `DecisionTreeClassifier()`
   - XGBoost: `XGBClassifier()`

6. **Evaluate**
   ```python
   accuracy = accuracy_score(y_test, y_pred)
   roc_auc = roc_auc_score(y_test, y_pred_proba)
   ```

7. **Save**
   - Models → `models/classification/*.pkl`
   - Metrics → `results/classification/metrics.json`

---

### Clustering Training (`train_clustering.py`)

**Process:**
1. **Load & Engineer Features** (same as classification)

2. **Scale Features**
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   joblib.dump(scaler, 'models/preprocessing/scaler.pkl')
   ```

3. **Train Models**
   - K-Means: `KMeans(n_clusters=5)`
   - DBSCAN: `DBSCAN(eps=0.5, min_samples=5)`

4. **Evaluate**
   ```python
   silhouette = silhouette_score(X, labels)
   ```

5. **Save**
   - Models → `models/clustering/*.pkl`
   - Metrics → `results/clustering/silhouette_score.json`

---

### Forecasting Training (`train_forecasting.py`)

**Process:**
1. **Load Data & Detect Time Series**
   ```python
   df = load_processed_data()
   datetime_col = detect_datetime_column(df)
   value_col = detect_value_column(df)
   ```

2. **Create Time Series**
   ```python
   ts_df = df.groupby(datetime_col)[value_col].sum().reset_index()
   ts_df.columns = ['ds', 'y']
   ```

3. **Split Train/Test**
   ```python
   train_size = int(len(ts_df) * 0.8)
   train = ts_df[:train_size]
   test = ts_df[train_size:]
   ```

4. **Train Models**
   - ARIMA: `ARIMA(train, order=(1,1,1))`
   - Prophet: `Prophet().fit(train)`

5. **Evaluate**
   ```python
   rmse = np.sqrt(mean_squared_error(test, forecast))
   mae = mean_absolute_error(test, forecast)
   ```

6. **Save**
   - Models → `models/forecasting/*.pkl`
   - Metrics → `results/forecasting/rmse.json`

---

## Model Types & Algorithms

### 1. Classification Models

**Purpose:** Predict next zone/location visit

| Model | Algorithm | Use Case | File |
|-------|-----------|----------|------|
| Random Forest | Ensemble of Decision Trees | Best overall performance | `zone_rf.pkl` |
| Decision Tree | Single Decision Tree | Baseline, interpretable | `baseline_dt.pkl` |
| XGBoost | Gradient Boosting | High performance | `zone_xgb.pkl` |

**Target Variable:** Zone/Location (categorical)

**Features:** All numeric engineered features (103 features)

---

### 2. Clustering Models

**Purpose:** Group customers with similar movement patterns

| Model | Algorithm | Use Case | File |
|-------|-----------|----------|------|
| K-Means | Centroid-based | Fixed number of clusters | `kmeans.pkl` |
| DBSCAN | Density-based | Variable clusters, noise detection | `dbscan.pkl` |

**Features:** All numeric features (scaled)

**Output:** Cluster labels for each customer

---

### 3. Forecasting Models

**Purpose:** Predict future traffic patterns

| Model | Algorithm | Use Case | File |
|-------|-----------|----------|------|
| ARIMA | AutoRegressive Integrated Moving Average | Statistical forecasting | `arima.pkl` |
| Prophet | Facebook's Prophet | Time series with seasonality | `prophet_model.pkl` |

**Input:** Time series data (datetime + value)

**Output:** Future predictions

---

## File Dependencies

### Dependency Graph

```
training/train_classification.py
    ├── streamlit_app/utils/data_loader.py
    │   └── data/processed/merged data set.csv
    ├── features/feature_engineering.py
    │   └── features/feature_config.yaml
    ├── sklearn (external)
    └── joblib (external)
        ↓
    models/classification/*.pkl
    results/classification/metrics.json
```

### Import Chain

```python
# training/train_classification.py
from streamlit_app.utils.data_loader import load_processed_data
    # → Loads: data/processed/merged data set.csv

from features.feature_engineering import FeatureEngineer
    # → Uses: features/feature_config.yaml
    # → Creates: engineered features

from sklearn.ensemble import RandomForestClassifier
    # → External library

import joblib
    # → Saves to: models/classification/zone_rf.pkl
    # → Saves to: results/classification/metrics.json
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    INPUT DATA                                │
│  data/processed/merged data set.csv                         │
│  Shape: (15,839 rows × 80 columns)                          │
│  Missing: 79,195 values                                      │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                            │
│  features/feature_engineering.py                             │
│  ├── Missing values → 0 missing                             │
│  ├── Datetime features → +20 features                       │
│  ├── Encoding → Categorical handled                         │
│  ├── Outliers → Capped                                      │
│  ├── Binning → Grouped features                             │
│  ├── Domain features → +5 features                         │
│  └── Combining → +5 features                               │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              ENGINEERED DATA                                 │
│  data/processed/engineered_features.csv                     │
│  Shape: (15,839 rows × 110 columns)                         │
│  Missing: 0 values                                           │
│  Numeric: 103 features                                      │
└──────────────────────┬───────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ CLASSIFICATION│ │  CLUSTERING  │ │  FORECASTING │
│              │ │              │ │              │
│ train_class  │ │ train_clust  │ │ train_fore   │
│              │ │              │ │              │
│ • RF         │ │ • K-Means    │ │ • ARIMA      │
│ • DT         │ │ • DBSCAN     │ │ • Prophet    │
│ • XGBoost    │ │              │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   MODELS     │ │   MODELS     │ │   MODELS     │
│              │ │              │ │              │
│ zone_rf.pkl  │ │ kmeans.pkl   │ │ arima.pkl    │
│ baseline_dt  │ │ dbscan.pkl   │ │ prophet.pkl  │
│ zone_xgb.pkl │ │              │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   RESULTS    │ │   RESULTS    │ │   RESULTS    │
│              │ │              │ │              │
│ metrics.json │ │ silhouette    │ │ rmse.json    │
│ confusion    │ │ cluster_plot  │ │ forecast     │
│ roc_auc      │ │               │ │              │
└──────────────┘ └───────────────┘ └──────────────┘
```

---

## Execution Order

### Recommended Sequence

1. **Verify Feature Engineering**
   ```bash
   python features/verify_feature_engineering.py
   ```

2. **Train Classification Models**
   ```bash
   python training/train_classification.py
   ```

3. **Train Clustering Models**
   ```bash
   python training/train_clustering.py
   ```

4. **Train Forecasting Models**
   ```bash
   python training/train_forecasting.py
   ```

### What Happens in Each Step

**Step 1: Verification**
- Checks if feature engineering is complete
- Verifies engineered features file exists
- Validates data quality

**Step 2-4: Training**
- Loads processed data
- Applies feature engineering (if not done)
- Trains models
- Evaluates performance
- Saves models and results

---

## Key Takeaways

1. **Feature Engineering is Central**
   - All training scripts use `FeatureEngineer`
   - Ensures consistent feature transformation

2. **Modular Design**
   - Each component has clear responsibility
   - Easy to modify individual parts

3. **Automatic Workflow**
   - Training scripts handle feature engineering automatically
   - No manual intervention needed

4. **Consistent Storage**
   - Models in `models/` folders
   - Results in `results/` folders
   - Easy to track and compare

5. **Reusable Components**
   - `data_loader.py` used by all scripts
   - `FeatureEngineer` used by all training scripts
   - Models can be loaded by dashboard and API

---

## Summary

The ML training workflow in this project follows a clear, modular architecture:

1. **Data** → Stored in `data/processed/`
2. **Features** → Engineered by `features/feature_engineering.py`
3. **Training** → Executed by `training/train_*.py` scripts
4. **Models** → Saved to `models/` folders
5. **Results** → Stored in `results/` folders
6. **Usage** → Models loaded by dashboard and API

All components work together seamlessly, with feature engineering as the central transformation step that prepares data for all ML models.


