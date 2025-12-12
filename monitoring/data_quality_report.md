# Data Quality Report

This document describes the data quality monitoring system for the Mall Movement Tracking project.

## Overview

The data quality monitoring system tracks and reports on various aspects of data quality to ensure reliable model performance and accurate predictions.

## Quality Metrics

### 1. Completeness
- **Total Cells**: Total number of data cells
- **Missing Cells**: Number of missing values
- **Completeness Rate**: Percentage of non-missing values
- **Missing Per Column**: Breakdown of missing values by column

### 2. Validity
- **Numeric Columns**: Count of numeric columns
- **Infinite Values**: Count of infinite values
- **Negative Values**: Columns with unexpected negative values

### 3. Consistency
- **Duplicate Rows**: Number of duplicate rows
- **Duplicate Rate**: Percentage of duplicate rows

### 4. Uniqueness
- **Unique Values Per Column**: Count of unique values for each column

### 5. Accuracy
- **Outliers Detected**: Number of outliers per column (IQR method)
- **Data Range**: Min, max, mean, std for numeric columns

## Quality Score

Overall quality score (0-100) calculated based on:
- Completeness rate
- Duplicate rate
- Presence of infinite values
- Data consistency

## Usage

### Generate Quality Report

```python
from monitoring.data_quality import DataQualityMonitor
from streamlit_app.utils.data_loader import load_processed_data

# Load data
df = load_processed_data()

# Initialize monitor
monitor = DataQualityMonitor()

# Generate report
report = monitor.generate_quality_report(df, output_path="data_quality_report.json")
```

### Track Quality Over Time

```python
# Track metrics
monitor.track_quality_over_time(df)

# Get trends
trends = monitor.get_quality_trends()
```

## Recommendations

The system automatically generates recommendations based on detected issues:

- **High Severity**: Critical issues requiring immediate attention
- **Medium Severity**: Issues that should be addressed
- **Low Severity**: Minor issues for monitoring

## Report Format

Quality reports are saved as JSON files with the following structure:

```json
{
  "report_type": "Data Quality Report",
  "generated_at": "2024-12-11 12:00:00",
  "metrics": {
    "completeness": {...},
    "validity": {...},
    "consistency": {...},
    "quality_score": 95.5
  },
  "recommendations": [...]
}
```

## Integration

The data quality monitoring can be integrated into:
- Automated data pipelines
- Model training workflows
- Streamlit dashboard
- Scheduled reports

## Best Practices

1. **Regular Monitoring**: Run quality checks regularly (daily/weekly)
2. **Thresholds**: Set appropriate thresholds for your use case
3. **Alerting**: Set up alerts for quality score drops
4. **Documentation**: Document quality issues and resolutions
5. **Trend Analysis**: Track quality trends over time

---

**Last Updated**: 2024-12-11

