# Architecture Diagram Plan

## Overview

This document outlines the plan for creating a comprehensive architecture diagram for the Mall Movement Tracking ML project.

## Diagram Components

### 1. Data Layer (Bottom)
- **Raw Data**: `data/processed/merged data set.csv`
- **Engineered Data**: `data/processed/engineered_features.csv`
- **Sample Data**: `data/sample/`

### 2. Feature Engineering Layer
- **FeatureEngineer Class**: `features/feature_engineering.py`
- **Configuration**: `features/feature_config.yaml`
- **Pipeline Steps**:
  - Missing Value Handling
  - Datetime Extraction
  - Categorical Encoding
  - Outlier Detection
  - Domain Features
  - Binning/Grouping

### 3. Model Training Layer
- **Classification Models**:
  - Random Forest (`models/classification/zone_rf.pkl`)
  - Decision Tree (`models/classification/baseline_dt.pkl`)
  - XGBoost (`models/classification/zone_xgb.pkl`)
  - SVM (`models/classification/zone_svm.pkl`)
- **Clustering Models**:
  - K-Means (`models/clustering/kmeans.pkl`)
  - DBSCAN (`models/clustering/dbscan.pkl`)
- **Forecasting Models**:
  - ARIMA (`models/forecasting/arima.pkl`)
  - Prophet (`models/forecasting/prophet_model.pkl`)

### 4. Results & Metrics Layer
- **Classification Results**: `results/classification/`
- **Clustering Results**: `results/clustering/`
- **Forecasting Results**: `results/forecasting/`

### 5. Application Layer (Top)
- **Streamlit Dashboard**: `streamlit_app/`
  - 8 Pages (Overview, Data Explorer, Heatmaps, etc.)
- **FastAPI**: `api/app.py`
  - RESTful endpoints
  - Model serving

### 6. Supporting Components
- **Monitoring**: `monitoring/`
  - Data Quality Monitoring
  - Drift Detection
- **Testing**: `tests/`
  - Unit tests
  - Integration tests
- **Notebooks**: `notebooks/`
  - EDA, Feature Analysis, Modeling, Comparison

## Data Flow

```
Raw Data → Feature Engineering → Model Training → Results → Applications
                ↓                      ↓              ↓
            Config Files          Model Files    Metrics/Plots
```

## Diagram Layout

### Horizontal Flow (Left to Right)
1. **Input**: Data sources
2. **Processing**: Feature engineering
3. **ML Models**: Training and inference
4. **Output**: Results and applications

### Vertical Layers (Bottom to Top)
1. **Data Layer** (Bottom)
2. **Processing Layer**
3. **Model Layer**
4. **Application Layer** (Top)

## Visual Elements

### Colors
- **Data Layer**: Blue (#3B82F6)
- **Feature Engineering**: Green (#10B981)
- **Model Training**: Orange (#F59E0B)
- **Results**: Purple (#8B5CF6)
- **Applications**: Red (#EF4444)
- **Supporting**: Gray (#6B7280)

### Shapes
- **Rectangles**: Components/Modules
- **Cylinders**: Data Storage
- **Arrows**: Data Flow
- **Dashed Lines**: Optional/Supporting

### Text Labels
- Component names
- File paths (optional)
- Data flow directions

## Diagram Specifications

### Size
- **Width**: 1920px (Full HD)
- **Height**: 1080px (Full HD)
- **DPI**: 300 (for print quality)

### Format
- **File Format**: PNG
- **Background**: White or transparent
- **Font**: Arial or similar, size 10-12pt

## Components to Include

### Main Components
1. ✅ Data Sources
2. ✅ Feature Engineering Pipeline
3. ✅ Model Training Scripts
4. ✅ Trained Models
5. ✅ Results Storage
6. ✅ Streamlit Dashboard
7. ✅ FastAPI
8. ✅ Monitoring
9. ✅ Testing

### Connections
- Data → Feature Engineering
- Feature Engineering → Model Training
- Model Training → Trained Models
- Trained Models → Results
- Trained Models → Applications
- Results → Applications
- Supporting components → All layers

## Implementation Options

### Option 1: Python Script (matplotlib + networkx)
- **Pros**: Programmatic, version-controlled, easy to update
- **Cons**: Requires coding

### Option 2: Graphviz/DOT
- **Pros**: Clean, professional, easy to maintain
- **Cons**: Requires Graphviz installation

### Option 3: Manual Design Tool
- **Pros**: Full control, custom design
- **Cons**: Not version-controlled, harder to update

## Recommended Approach

**Use Python script with matplotlib** for:
- Version control
- Easy updates
- Consistent styling
- Automation

---

## Next Steps

1. ✅ Create plan document (this file)
2. ⏳ Create Python script to generate diagram
3. ⏳ Generate initial diagram
4. ⏳ Review and refine
5. ⏳ Save as PNG in `docs/architecture_diagram.png`


