# Streamlit Dashboard Documentation

## Overview

The Streamlit dashboard provides a production-ready, user-friendly interface for visualizing ML model results, exploring data, and making predictions. The dashboard features a modern, clean design with a light theme (white sidebar, light background).

## Architecture

### File Structure

```
streamlit_app/
├── app.py                    # Main application entry point
├── pages/                     # Streamlit pages (auto-routed)
│   ├── 1_Overview.py         # Dashboard home page
│   ├── 2_Data_Explorer.py    # Data exploration tools
│   ├── 3_Heatmaps.py         # Movement pattern visualizations
│   ├── 4_Classification_Results.py  # Classification model metrics
│   ├── 5_Clustering_Insights.py     # Clustering analysis
│   ├── 6_Forecasting_Traffic.py     # Forecasting models
│   ├── 7_Predict_Next_Zone.py       # Prediction interface
│   └── 8_Model_Explainability.py    # Feature importance
├── utils/                     # Utility modules
│   ├── data_loader.py         # Data loading functions
│   ├── model_loader.py        # Model loading functions
│   ├── charts.py             # Visualization utilities
│   └── preprocess.py          # Preprocessing functions
└── assets/                    # Static assets
    ├── colors.css            # Custom styling
    └── logo.png              # Logo image
```

## Pages

### 1. Overview (Home Page)
- **Purpose**: Dashboard home with key metrics and quick access
- **Features**:
  - Project overview metrics (records, features, models)
  - Key performance indicators
  - Model status summary
  - Navigation guide

### 2. Data Explorer
- **Purpose**: Interactive data exploration
- **Features**:
  - Dataset overview (rows, columns, missing values)
  - Data preview (first/last/random rows)
  - Column information and statistics
  - Distribution plots (histogram, box plot)

### 3. Heatmaps
- **Purpose**: Visualize movement patterns
- **Features**:
  - Zone popularity heatmap
  - Zone-user interaction heatmap
  - Temporal movement heatmap
  - Heatmap statistics

### 4. Classification Results
- **Purpose**: Display classification model performance
- **Features**:
  - Model comparison (Random Forest, Decision Tree, XGBoost)
  - Accuracy metrics and ROC-AUC scores
  - Confusion matrix visualization
  - Feature importance analysis

### 5. Clustering Insights
- **Purpose**: Customer segmentation analysis
- **Features**:
  - Clustering metrics (K-Means, DBSCAN)
  - Silhouette score comparison
  - Cluster distribution visualization
  - Cluster statistics

### 6. Forecasting Traffic
- **Purpose**: Traffic prediction models
- **Features**:
  - Forecasting metrics (RMSE, MAE)
  - Model comparison charts
  - Forecasting details table

### 7. Predict Next Zone
- **Purpose**: Real-time prediction interface
- **Features**:
  - Model selection (XGBoost, Random Forest, Decision Tree)
  - Single prediction form
  - Batch prediction (CSV upload)
  - Prediction probabilities

### 8. Model Explainability
- **Purpose**: Understand model decisions
- **Features**:
  - Feature importance analysis
  - Top features visualization
  - Model comparison insights

## Design Features

### Color Scheme
- **Primary Color**: #1f77b4 (Blue)
- **Background**: #f8f9fa (Light gray)
- **Sidebar**: #ffffff (White)
- **Text**: #2c3e50 (Dark gray)

### Styling
- Clean, modern interface
- Light theme (no black backgrounds)
- Responsive layout
- Professional metrics display
- Custom button styling

## Running the Dashboard

### Prerequisites
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### Start the Dashboard
```bash
cd streamlit_app
streamlit run app.py
```

Or from project root:
```bash
streamlit run streamlit_app/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Data Flow

1. **Data Loading**: `data_loader.py` loads processed data from `data/processed/merged data set.csv`
2. **Feature Engineering**: `FeatureEngineer` class applies feature transformations
3. **Model Loading**: `model_loader.py` loads trained models from `models/` directory
4. **Visualization**: Matplotlib/Seaborn creates charts and plots
5. **Display**: Streamlit renders all components

## Integration Points

### With Feature Engineering
- Uses `FeatureEngineer` class for real-time feature engineering
- Applies same transformations as training pipeline

### With Models
- Loads models from `models/` directory
- Uses same preprocessing (scaler, encoder) as training

### With Results
- Reads metrics from `results/` directory
- Displays saved visualizations

## Customization

### Adding New Pages
1. Create new file in `streamlit_app/pages/` with format `N_PageName.py`
2. Import required utilities
3. Add Streamlit components (st.title, st.header, etc.)
4. Streamlit will auto-detect and add to navigation

### Modifying Styling
- Edit `streamlit_app/app.py` CSS section
- Or modify `streamlit_app/assets/colors.css`

### Adding New Visualizations
- Use `streamlit_app/utils/charts.py` helper functions
- Or create custom matplotlib/seaborn plots

## Best Practices

1. **Error Handling**: All pages include try-except blocks
2. **Loading States**: Use `st.spinner()` for long operations
3. **Data Caching**: Consider `@st.cache_data` for expensive operations
4. **Responsive Design**: Use `st.columns()` for layouts
5. **User Feedback**: Use `st.success()`, `st.error()`, `st.warning()`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure project root is in Python path
2. **File Not Found**: Check data/model file paths
3. **Memory Issues**: Use data sampling for large datasets
4. **Slow Loading**: Implement caching with `@st.cache_data`

## Future Enhancements

- [ ] Add data caching for better performance
- [ ] Implement user authentication
- [ ] Add export functionality for charts
- [ ] Real-time data updates
- [ ] Advanced filtering options
- [ ] Model retraining interface

