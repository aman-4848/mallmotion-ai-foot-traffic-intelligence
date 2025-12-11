# Mall Movement Tracking - Workflow

## Feature Engineering Workflow

### Step 1: Run Feature Engineering (BEFORE ML Training)

Feature engineering must be run before training any ML models.

```bash
# Option 1: Run feature engineering pipeline
python features/run_feature_engineering.py

# Option 2: Run feature analysis (includes engineering + analysis)
python features/feature_analysis.py
```

This will:
- Load processed data from `data/processed/merged data set.csv`
- Apply comprehensive feature engineering
- Save engineered features to `data/processed/engineered_features.csv`
- Create analysis visualizations in `results/feature_analysis/`

### Step 2: Train Models

After feature engineering, train each model:

```bash
# Train classification models
python training/train_classification.py

# Train clustering models
python training/train_clustering.py

# Train forecasting models
python training/train_forecasting.py
```

**Note:** All training scripts automatically apply feature engineering if you haven't run it separately. They use the `FeatureEngineer` class internally.

### Step 3: Use API

Start the API to view results:

```bash
cd api
uvicorn app:app --reload
```

## Feature Engineering Components

The feature engineering pipeline includes:

1. ✅ **Missing Value Handling** - Automatic imputation
2. ✅ **Categorical Encoding** - Label & one-hot encoding
3. ✅ **Datetime Extraction** - Comprehensive temporal features
4. ✅ **Outlier Detection** - IQR and Z-score methods
5. ✅ **Binning/Grouping** - Quantile and uniform binning
6. ✅ **Domain Features** - Mall movement specific features
7. ✅ **Column Combining** - Automatic feature combinations

## Data Flow

```
Processed Data (merged data set.csv)
    ↓
Feature Engineering Pipeline
    ↓
Engineered Features (engineered_features.csv)
    ↓
ML Training (Classification, Clustering, Forecasting)
    ↓
Trained Models + Results
    ↓
API / Streamlit Dashboard
```

## Configuration

Edit `features/feature_config.yaml` to customize feature engineering behavior.

