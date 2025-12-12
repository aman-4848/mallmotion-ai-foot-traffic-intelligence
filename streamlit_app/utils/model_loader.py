"""
Model Loader Utility
Loads trained models for predictions
"""
import joblib
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

def load_classification_model(model_name='xgboost', return_scaler=False):
    """
    Load classification model
    
    Args:
        model_name: 'xgboost', 'random_forest', 'decision_tree', or 'logistic_regression'
        return_scaler: If True, also return the scaler (for Logistic Regression)
    
    Returns:
        Trained model (and scaler if return_scaler=True)
    """
    model_files = {
        'xgboost': 'classification/zone_xgb.pkl',
        'random_forest': 'classification/zone_rf.pkl',
        'decision_tree': 'classification/baseline_dt.pkl',
        'logistic_regression': 'classification/zone_lr.pkl'
    }
    
    model_path = MODELS_DIR / model_files.get(model_name.lower(), model_files['xgboost'])
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Load scaler for Logistic Regression if requested
    if return_scaler and model_name.lower() == 'logistic_regression':
        scaler_path = MODELS_DIR / 'preprocessing' / 'lr_scaler.pkl'
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            return model, scaler
        else:
            return model, None
    
    return model

def load_clustering_model(model_name='kmeans'):
    """
    Load clustering model
    
    Args:
        model_name: 'kmeans' or 'dbscan'
    
    Returns:
        Trained model
    """
    model_files = {
        'kmeans': 'clustering/kmeans.pkl',
        'dbscan': 'clustering/dbscan.pkl'
    }
    
    model_path = MODELS_DIR / model_files.get(model_name.lower(), model_files['kmeans'])
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return joblib.load(model_path)

def load_forecasting_model(model_name='random_forest_regressor'):
    """
    Load forecasting model
    
    Args:
        model_name: 'random_forest_regressor' or 'arima'
    
    Returns:
        Trained model
    """
    model_files = {
        'random_forest_regressor': 'forecasting/rf_forecast.pkl',
        'rf_forecast': 'forecasting/rf_forecast.pkl',
        'arima': 'forecasting/arima.pkl'
    }
    
    model_path = MODELS_DIR / model_files.get(model_name.lower(), model_files['random_forest_regressor'])
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return joblib.load(model_path)

def load_preprocessor(preprocessor_name='encoder'):
    """
    Load preprocessing objects
    
    Args:
        preprocessor_name: 'encoder' or 'scaler'
    
    Returns:
        Preprocessing object
    """
    preprocessor_files = {
        'encoder': 'preprocessing/encoder.pkl',
        'scaler': 'preprocessing/scaler.pkl',
        'forecast_scaler': 'preprocessing/forecast_scaler.pkl',
        'forecast_features': 'preprocessing/forecast_features.pkl'
    }
    
    preprocessor_path = MODELS_DIR / preprocessor_files.get(preprocessor_name.lower(), preprocessor_files['encoder'])
    
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
    
    return joblib.load(preprocessor_path)

