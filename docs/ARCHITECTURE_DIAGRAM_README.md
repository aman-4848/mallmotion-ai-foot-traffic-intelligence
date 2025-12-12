# Architecture Diagram

## Overview

The architecture diagram provides a visual representation of the Mall Movement Tracking ML project's system architecture, showing data flow, components, and their relationships.

## File Location

- **Diagram**: `docs/architecture_diagram.png`
- **Generator Script**: `docs/generate_architecture_diagram.py`
- **Plan Document**: `docs/ARCHITECTURE_DIAGRAM_PLAN.md`

## Diagram Components

### 1. Data Layer (Bottom)
- Raw data sources
- Engineered features
- Sample data

### 2. Feature Engineering Layer
- Feature engineering pipeline
- Configuration files
- Processing steps (missing values, datetime, encoding, etc.)

### 3. Model Training Layer
- Classification models (RF, DT, XGBoost, SVM)
- Clustering models (K-Means, DBSCAN)
- Forecasting models (ARIMA, Prophet)

### 4. Results Layer
- Classification results
- Clustering results
- Forecasting results

### 5. Application Layer (Top)
- Streamlit Dashboard
- FastAPI REST API

### 6. Supporting Components
- Monitoring (data quality, drift detection)
- Testing (unit tests, integration tests)
- Notebooks (EDA, analysis, experiments)

## Color Scheme

- **Blue**: Data Layer
- **Green**: Feature Engineering
- **Orange**: Model Training
- **Purple**: Results
- **Red**: Applications
- **Gray**: Supporting Components

## Regenerating the Diagram

To regenerate the architecture diagram:

```bash
python docs/generate_architecture_diagram.py
```

The script will create/update `docs/architecture_diagram.png` with:
- **Resolution**: 1920x1080 pixels
- **DPI**: 300 (print quality)
- **Format**: PNG

## Customization

Edit `docs/generate_architecture_diagram.py` to:
- Add new components
- Change colors
- Modify layout
- Update labels

## Usage

The diagram can be used for:
- Project documentation
- Presentations
- Architecture reviews
- Onboarding new team members
- Project proposals

---

**Last Updated**: 2024-12-12


