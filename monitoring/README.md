# Monitoring Module

This module provides data quality monitoring and drift detection capabilities for the Mall Movement Tracking project.

## Modules

### 1. Data Quality Monitoring (`data_quality.py`)

Monitors data quality metrics and generates comprehensive reports.

**Features:**
- Completeness tracking (missing values)
- Validity checks (infinite values, negative values)
- Consistency checks (duplicates)
- Uniqueness analysis
- Accuracy metrics (outliers, data ranges)
- Overall quality score calculation
- Automated recommendations

**Usage:**
```python
from monitoring.data_quality import DataQualityMonitor
from streamlit_app.utils.data_loader import load_processed_data

# Load data
df = load_processed_data()

# Initialize monitor
monitor = DataQualityMonitor()

# Generate report
report = monitor.generate_quality_report(df, output_path="quality_report.json")

# Track over time
monitor.track_quality_over_time(df)
trends = monitor.get_quality_trends()
```

**Output:**
- JSON report with all quality metrics
- Quality score (0-100)
- Recommendations for improvement

### 2. Drift Detection (`drift_detection.py`)

Detects changes in data distribution over time using multiple statistical methods.

**Features:**
- Kolmogorov-Smirnov test for distribution comparison
- Statistical comparison (mean/std differences)
- Population Stability Index (PSI)
- Comprehensive drift reports

**Usage:**
```python
from monitoring.drift_detection import DriftDetector
from streamlit_app.utils.data_loader import load_processed_data

# Load reference and current data
df = load_processed_data()
reference_data = df.iloc[:len(df)//2]
current_data = df.iloc[len(df)//2:]

# Initialize detector
detector = DriftDetector(reference_data)

# Generate drift report
report = detector.generate_drift_report(current_data, output_path="drift_report.json")
```

**Detection Methods:**
1. **KS Test**: Statistical test for distribution differences
2. **Statistical Comparison**: Mean and standard deviation comparison
3. **PSI**: Population Stability Index for feature stability

**Output:**
- Drift detection results per method
- List of drifted features
- Drift severity assessment
- Recommendations

## Quick Start

### Run Data Quality Check

```bash
python monitoring/data_quality.py
```

### Run Drift Detection

```bash
python monitoring/drift_detection.py
```

## Integration

### Streamlit Dashboard

Add monitoring widgets to Streamlit pages:

```python
from monitoring.data_quality import DataQualityMonitor

monitor = DataQualityMonitor()
report = monitor.generate_quality_report(df)

st.metric("Quality Score", f"{report['metrics']['quality_score']:.1f}/100")
```

### Automated Pipeline

Integrate into data processing pipeline:

```python
# After data loading
monitor = DataQualityMonitor()
quality_report = monitor.generate_quality_report(new_data)

if quality_report['metrics']['quality_score'] < 70:
    # Alert or stop pipeline
    raise ValueError("Data quality below threshold")
```

## Report Files

- `data_quality_report.json` - Quality metrics and recommendations
- `drift_report.json` - Drift detection results

## Dependencies

- pandas
- numpy
- scipy (optional, for KS test)
- json

## Best Practices

1. **Regular Monitoring**: Run quality checks on new data
2. **Baseline Establishment**: Use initial data as reference
3. **Threshold Setting**: Define acceptable quality/drift thresholds
4. **Alerting**: Set up alerts for quality degradation
5. **Documentation**: Document quality issues and resolutions

---

**Last Updated**: 2024-12-11

