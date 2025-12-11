# Feature Engineering Module

This module provides comprehensive feature engineering capabilities for the mall movement tracking project.

## Overview

The feature engineering pipeline includes:

1. **Missing Value Handling** - Automatic imputation based on data type
2. **Datetime Feature Extraction** - Comprehensive temporal features
3. **Categorical Encoding** - Label encoding and one-hot encoding
4. **Outlier Detection & Handling** - IQR and Z-score methods
5. **Binning/Grouping** - Quantile and uniform binning
6. **Domain-Specific Features** - Mall movement tracking features
7. **Column Combining** - Automatic feature combinations

## Usage

### Quick Start

```python
from streamlit_app.utils.data_loader import load_processed_data
from features.feature_engineering import FeatureEngineer

# Load data
df = load_processed_data()

# Engineer features
fe = FeatureEngineer()
df_engineered = fe.engineer_features(df)
```

### Run Feature Engineering Pipeline

```bash
python features/run_feature_engineering.py
```

This will:
- Load processed data
- Apply all feature engineering steps
- Save engineered features to `data/processed/engineered_features.csv`

### Run Feature Analysis

```bash
python features/feature_analysis.py
```

This will:
- Analyze data before and after engineering
- Create visualizations
- Save analysis results to `results/feature_analysis/`

## Configuration

Edit `feature_config.yaml` to customize feature engineering:

```yaml
missing_values:
  strategy: "auto"  # auto, mean, median, mode, forward_fill, drop

encoding:
  method: "auto"  # auto, label, onehot
  columns: null  # null = auto-detect

outliers:
  method: "iqr"  # iqr, zscore
  columns: null

binning:
  enabled: true
  n_bins: 5
  method: "quantile"

domain:
  zone_column: null  # auto-detect
  user_column: null  # auto-detect
```

## Features Created

### Temporal Features
- `hour`, `day_of_week`, `day_of_month`, `month`, `year`, `quarter`
- `is_weekend`, `is_weekday`, `is_morning`, `is_afternoon`, `is_evening`, `is_night`
- Cyclical encodings: `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, etc.

### Domain Features (Mall Movement)
- `visit_count` - Number of visits per user-zone combination
- `total_zones_visited` - Total unique zones visited per user
- `avg_visits_per_zone` - Average visits per zone per user
- `zone_popularity` - Total visits per zone across all users
- `user_activity_level` - Total visits per user

### Encoding
- Label encoding for high-cardinality categorical variables
- One-hot encoding for low-cardinality categorical variables

### Outlier Handling
- IQR method: Caps outliers at Q1 - 1.5*IQR and Q3 + 1.5*IQR
- Z-score method: Caps values beyond 3 standard deviations

## Integration with Training

All training scripts (`train_classification.py`, `train_clustering.py`, `train_forecasting.py`) automatically use feature engineering before model training.

## Files

- `feature_engineering.py` - Main feature engineering class
- `feature_config.yaml` - Configuration file
- `run_feature_engineering.py` - Standalone script to run feature engineering
- `feature_analysis.py` - Analysis and visualization script
- `feature_store.py` - Feature store utilities (if needed)

