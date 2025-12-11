# Feature Engineering Implementation Guide

## Overview

This document provides a comprehensive guide on how feature engineering is implemented in the Mall Movement Tracking ML project. Feature engineering is a critical step that transforms raw processed data into features suitable for machine learning models.

## Table of Contents

1. [Architecture](#architecture)
2. [Implementation Details](#implementation-details)
3. [Feature Engineering Pipeline](#feature-engineering-pipeline)
4. [Feature Categories](#feature-categories)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [Technical Details](#technical-details)
8. [Best Practices](#best-practices)

---

## Architecture

### Component Structure

```
features/
├── feature_engineering.py      # Main FeatureEngineer class
├── feature_config.yaml         # Configuration file
├── run_feature_engineering.py  # Standalone execution script
├── feature_analysis.py         # Analysis and visualization
└── verify_feature_engineering.py  # Verification script
```

### Class Design

The feature engineering is implemented using the `FeatureEngineer` class, which follows an object-oriented design pattern:

```python
class FeatureEngineer:
    def __init__(self, config_path=None)
    def detect_column_types(self, df)
    def handle_missing_values(self, df, strategy='auto')
    def extract_datetime_features(self, df, datetime_column=None)
    def encode_categorical_variables(self, df, columns=None, method='auto')
    def detect_and_handle_outliers(self, df, columns=None, method='iqr')
    def create_bins(self, df, columns=None, n_bins=5, method='quantile')
    def create_domain_features(self, df, zone_column=None, user_column=None)
    def combine_columns(self, df, combinations=None)
    def engineer_features(self, df, config=None)
```

---

## Implementation Details

### 1. FeatureEngineer Class

The `FeatureEngineer` class is the core component that orchestrates all feature engineering operations.

#### Initialization

```python
from features.feature_engineering import FeatureEngineer

# Initialize with default configuration
fe = FeatureEngineer()

# Initialize with custom configuration
fe = FeatureEngineer(config_path='features/feature_config.yaml')
```

**Key Attributes:**
- `config`: Configuration dictionary loaded from YAML
- `label_encoders`: Dictionary storing LabelEncoder instances
- `onehot_encoders`: Dictionary storing OneHotEncoder instances
- `imputers`: Dictionary storing SimpleImputer instances
- `scalers`: Dictionary storing StandardScaler instances
- `bin_edges`: Dictionary storing bin edge information

#### Column Type Detection

The system automatically detects column types:

```python
column_types = fe.detect_column_types(df)
# Returns:
# {
#     'datetime': ['TIMESTAMP', 'modificationTime'],
#     'categorical': [],
#     'numeric': ['USERID', 'WAP011', ...],
#     'id': [],
#     'target': []
# }
```

**Detection Logic:**
- **Datetime**: Columns with `datetime64` dtype or names containing 'date', 'time', 'timestamp'
- **Categorical**: Object dtype columns
- **Numeric**: Integer and float columns
- **ID**: Columns with 'id', 'user', 'customer' in name
- **Target**: Columns with 'zone', 'location' in name

---

## Feature Engineering Pipeline

The feature engineering follows a sequential pipeline with 7 main steps:

### Step 1: Missing Value Handling

**Implementation:**
```python
df = fe.handle_missing_values(df, strategy='auto')
```

**Strategies:**
- `auto`: Automatically chooses strategy based on data type
  - Numeric columns → Median imputation
  - Categorical columns → Mode imputation
- `mean`: Mean imputation for numeric columns
- `median`: Median imputation for numeric columns
- `mode`: Mode imputation for categorical columns
- `forward_fill`: Forward fill then backward fill
- `drop`: Drop rows with missing values

**Technical Details:**
- Uses `SimpleImputer` from scikit-learn
- Handles edge cases (all NaN columns)
- Stores imputers for later use on test data

**Example:**
```python
# Before: 79,195 missing values
# After: 0 missing values
```

### Step 2: Datetime Feature Extraction

**Implementation:**
```python
df = fe.extract_datetime_features(df, datetime_column='TIMESTAMP')
```

**Features Created:**
- **Basic temporal features:**
  - `hour`: Hour of day (0-23)
  - `day_of_week`: Day of week (0-6)
  - `day_of_month`: Day of month (1-31)
  - `day_of_year`: Day of year (1-365)
  - `week`: Week number (1-52)
  - `month`: Month (1-12)
  - `quarter`: Quarter (1-4)
  - `year`: Year

- **Boolean temporal features:**
  - `is_weekend`: 1 if Saturday/Sunday, else 0
  - `is_weekday`: 1 if Monday-Friday, else 0
  - `is_morning`: 1 if 6-11 AM, else 0
  - `is_afternoon`: 1 if 12-5 PM, else 0
  - `is_evening`: 1 if 6-11 PM, else 0
  - `is_night`: 1 if 12-5 AM, else 0

- **Cyclical encodings:**
  - `hour_sin`, `hour_cos`: Cyclical encoding of hour
  - `day_of_week_sin`, `day_of_week_cos`: Cyclical encoding of day
  - `month_sin`, `month_cos`: Cyclical encoding of month

**Why Cyclical Encoding?**
- Captures periodic patterns (e.g., hour 23 is close to hour 0)
- Improves model performance for temporal features

### Step 3: Categorical Encoding

**Implementation:**
```python
df = fe.encode_categorical_variables(df, method='auto')
```

**Methods:**
- `auto`: 
  - High cardinality (>10 unique values) → Label encoding
  - Low cardinality (≤10 unique values) → One-hot encoding
- `label`: Label encoding for all categorical columns
- `onehot`: One-hot encoding for all categorical columns

**Label Encoding:**
- Maps categorical values to integers
- Example: ['Zone A', 'Zone B', 'Zone C'] → [0, 1, 2]
- Stored in `self.label_encoders` for consistency

**One-Hot Encoding:**
- Creates binary columns for each category
- Example: `zone_Zone_A`, `zone_Zone_B`, `zone_Zone_C`
- Drops first category to avoid multicollinearity

### Step 4: Outlier Detection & Handling

**Implementation:**
```python
df = fe.detect_and_handle_outliers(df, method='iqr')
```

**Methods:**

1. **IQR Method (Interquartile Range):**
   ```python
   Q1 = df[col].quantile(0.25)
   Q3 = df[col].quantile(0.75)
   IQR = Q3 - Q1
   lower_bound = Q1 - 1.5 * IQR
   upper_bound = Q3 + 1.5 * IQR
   df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
   ```

2. **Z-Score Method:**
   ```python
   z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
   # Cap values beyond 3 standard deviations
   df.loc[z_scores > 3, col] = df[col].mean() + 3 * np.sign(...) * df[col].std()
   ```

**Why Handle Outliers?**
- Prevents models from being skewed by extreme values
- Improves model stability and generalization

### Step 5: Binning/Grouping

**Implementation:**
```python
df = fe.create_bins(df, n_bins=5, method='quantile')
```

**Methods:**
- `quantile`: Equal number of observations per bin
- `uniform`: Equal width bins

**Use Cases:**
- Reduces noise in continuous variables
- Creates categorical features from numeric ones
- Helps with non-linear relationships

**Example:**
```python
# Original: Continuous values 0-100
# Binned: 0, 1, 2, 3, 4 (5 bins)
```

### Step 6: Domain-Specific Features

**Implementation:**
```python
df = fe.create_domain_features(df, zone_column='SPACEID', user_column='USERID')
```

**Features Created:**

1. **Visit Count:**
   ```python
   visit_count = df.groupby([user_column, zone_column]).size()
   ```

2. **Total Zones Visited:**
   ```python
   total_zones_visited = df.groupby(user_column)[zone_column].nunique()
   ```

3. **Average Visits per Zone:**
   ```python
   avg_visits_per_zone = visit_count / total_zones_visited
   ```

4. **Zone Popularity:**
   ```python
   zone_popularity = df.groupby(zone_column).size()
   ```

5. **User Activity Level:**
   ```python
   user_activity_level = df.groupby(user_column).size()
   ```

**Why Domain Features?**
- Captures business logic and domain knowledge
- Improves model interpretability
- Often highly predictive

### Step 7: Column Combining

**Implementation:**
```python
df = fe.combine_columns(df, combinations=None)
```

**Automatic Combinations:**
- Sum: `col1_plus_col2 = col1 + col2`
- Multiply: `col1_mult_col2 = col1 * col2`
- Divide: `col1_div_col2 = col1 / col2` (with zero handling)
- Subtract: `col1_minus_col2 = col1 - col2`

**Custom Combinations:**
```python
combinations = [
    ('LONGITUDE', 'LATITUDE', 'multiply', 'location_product'),
    ('visit_count', 'total_zones_visited', 'divide', 'avg_zone_visits')
]
df = fe.combine_columns(df, combinations=combinations)
```

---

## Feature Categories

### Summary of Features Created

Based on the project's data (15,839 rows, 80 original columns):

| Category | Count | Examples |
|----------|-------|----------|
| **Original Features** | 80 | USERID, TIMESTAMP, WAP011, LONGITUDE, LATITUDE |
| **Temporal Features** | ~20 | hour, day_of_week, month, is_weekend, hour_sin, hour_cos |
| **Domain Features** | ~5 | visit_count, total_zones_visited, zone_popularity |
| **Binned Features** | Variable | WAP011_binned, LONGITUDE_binned |
| **Combined Features** | Variable | WAP011_plus_WAP012, LONGITUDE_mult_LATITUDE |
| **Encoded Features** | Variable | Categorical columns encoded |
| **Total Engineered** | 110 | 30 new features added |

---

## Configuration

### Configuration File: `feature_config.yaml`

```yaml
# Missing values handling
missing_values:
  strategy: "auto"  # auto, mean, median, mode, forward_fill, drop

# Datetime feature extraction
datetime:
  column: null  # null = auto-detect
  enabled: true

# Categorical encoding
encoding:
  method: "auto"  # auto, label, onehot
  columns: null  # null = auto-detect all categorical

# Outlier detection and handling
outliers:
  method: "iqr"  # iqr, zscore
  columns: null  # null = all numeric columns

# Binning/grouping
binning:
  enabled: true
  n_bins: 5
  method: "quantile"  # quantile, uniform
  columns: null  # null = auto-select numeric columns

# Domain-specific features
domain:
  zone_column: null  # null = auto-detect
  user_column: null  # null = auto-detect
  enabled: true

# Column combining
combining:
  enabled: true
  combinations: null  # null = auto-create
```

### Customizing Configuration

1. **Edit YAML file:**
   ```yaml
   missing_values:
     strategy: "median"  # Change from auto to median
   ```

2. **Use in code:**
   ```python
   fe = FeatureEngineer(config_path='custom_config.yaml')
   ```

---

## Usage Examples

### Example 1: Basic Usage

```python
from streamlit_app.utils.data_loader import load_processed_data
from features.feature_engineering import FeatureEngineer

# Load data
df = load_processed_data()

# Initialize feature engineer
fe = FeatureEngineer()

# Engineer features
df_engineered = fe.engineer_features(df)

print(f"Original: {df.shape}")
print(f"Engineered: {df_engineered.shape}")
```

### Example 2: Step-by-Step Execution

```python
from features.feature_engineering import FeatureEngineer

fe = FeatureEngineer()
df = load_processed_data()

# Step 1: Handle missing values
df = fe.handle_missing_values(df, strategy='auto')

# Step 2: Extract datetime features
df = fe.extract_datetime_features(df, datetime_column='TIMESTAMP')

# Step 3: Encode categorical variables
df = fe.encode_categorical_variables(df, method='auto')

# Step 4: Handle outliers
df = fe.detect_and_handle_outliers(df, method='iqr')

# Step 5: Create bins
df = fe.create_bins(df, n_bins=5, method='quantile')

# Step 6: Domain features
df = fe.create_domain_features(df, zone_column='SPACEID', user_column='USERID')

# Step 7: Combine columns
df = fe.combine_columns(df)
```

### Example 3: Custom Configuration

```python
import yaml

# Create custom config
custom_config = {
    'missing_values': {'strategy': 'median'},
    'encoding': {'method': 'label'},
    'outliers': {'method': 'zscore'},
    'binning': {'enabled': False}
}

# Save to file
with open('custom_config.yaml', 'w') as f:
    yaml.dump(custom_config, f)

# Use custom config
fe = FeatureEngineer(config_path='custom_config.yaml')
df_engineered = fe.engineer_features(df)
```

### Example 4: Integration with Training

```python
# In training scripts
from features.feature_engineering import FeatureEngineer

def prepare_data():
    df = load_processed_data()
    
    # Apply feature engineering
    fe = FeatureEngineer()
    df = fe.engineer_features(df)
    
    # Prepare for model
    X = df.select_dtypes(include=[np.number])
    y = df['target_column']
    
    return X, y
```

---

## Technical Details

### Data Flow

```
Raw Processed Data (merged data set.csv)
    ↓
[Feature Engineering Pipeline]
    ├── Missing Value Handling
    ├── Datetime Extraction
    ├── Categorical Encoding
    ├── Outlier Handling
    ├── Binning
    ├── Domain Features
    └── Column Combining
    ↓
Engineered Features (engineered_features.csv)
    ↓
ML Models (Classification, Clustering, Forecasting)
```

### Memory Management

- All operations use `.copy()` to avoid modifying original data
- Large DataFrames are processed in-place where possible
- Imputers and encoders are stored for consistency

### Reproducibility

- All random operations use fixed seeds
- Transformers (imputers, encoders) are saved for test data
- Configuration is version-controlled

### Performance Considerations

- **Missing Value Handling**: O(n) where n is number of rows
- **Datetime Extraction**: O(n) - vectorized operations
- **Encoding**: O(n * m) where m is number of categories
- **Outlier Detection**: O(n) - vectorized operations
- **Binning**: O(n log n) for quantile-based binning
- **Domain Features**: O(n) with groupby operations

**Total Complexity**: Approximately O(n log n) for typical datasets

---

## Best Practices

### 1. Always Verify Before Training

```bash
python features/verify_feature_engineering.py
```

### 2. Save Engineered Features

```python
df_engineered.to_csv('data/processed/engineered_features.csv', index=False)
```

### 3. Use Consistent Configuration

- Use the same config for training and inference
- Version control your configuration files

### 4. Monitor Feature Quality

- Check for data leakage
- Verify feature distributions
- Ensure no target leakage

### 5. Document Custom Features

- Document domain-specific features
- Explain business logic
- Note any assumptions

### 6. Test on Sample Data First

```python
# Test on small sample
df_sample = df.head(1000)
df_engineered = fe.engineer_features(df_sample)
```

---

## Integration Points

### 1. Data Loading

```python
from streamlit_app.utils.data_loader import load_processed_data
```

### 2. Training Scripts

All training scripts automatically use feature engineering:
- `training/train_classification.py`
- `training/train_clustering.py`
- `training/train_forecasting.py`

### 3. Streamlit Dashboard

Features are loaded and used in dashboard pages:
- `streamlit_app/pages/2_Data_Explorer.py`
- `streamlit_app/pages/7_Predict_Next_Zone.py`

### 4. API Endpoints

API uses engineered features for predictions:
- `api/app.py`

---

## Troubleshooting

### Common Issues

1. **"No numeric features found"**
   - Solution: Check data types, ensure numeric columns exist

2. **"Memory error"**
   - Solution: Process in chunks or reduce dataset size

3. **"All NaN column"**
   - Solution: Feature engineering handles this automatically

4. **"Binning fails"**
   - Solution: Check for sufficient unique values (> n_bins)

5. **"Encoding fails"**
   - Solution: Ensure categorical columns are object type

---

## Future Enhancements

Potential improvements:

1. **Feature Selection**: Automatic feature importance-based selection
2. **Polynomial Features**: Create interaction terms automatically
3. **Target Encoding**: Mean encoding for categorical variables
4. **Feature Scaling**: Standardization/normalization options
5. **Feature Store**: Integration with feature store systems
6. **Pipeline Persistence**: Save entire pipeline for deployment

---

## References

- **Feature Engineering Module**: `features/feature_engineering.py`
- **Configuration**: `features/feature_config.yaml`
- **Usage Guide**: `features/README.md`
- **Training Guide**: `training/MODEL_TRAINING_GUIDE.md`
- **Workflow**: `WORKFLOW.md`

---

## Summary

Feature engineering in this project is implemented as a comprehensive, configurable pipeline that:

1. ✅ Handles missing values intelligently
2. ✅ Extracts rich temporal features
3. ✅ Encodes categorical variables appropriately
4. ✅ Detects and handles outliers
5. ✅ Creates bins for continuous variables
6. ✅ Generates domain-specific features
7. ✅ Combines columns for interactions

The implementation is:
- **Modular**: Each step is independent
- **Configurable**: YAML-based configuration
- **Reproducible**: Consistent transformations
- **Scalable**: Handles large datasets
- **Maintainable**: Well-documented code

This ensures that all models receive high-quality, well-engineered features that improve prediction performance.

