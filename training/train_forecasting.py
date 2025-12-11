"""
Train Forecasting Models
Trains ARIMA and Prophet forecasting models using processed data
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from streamlit_app.utils.data_loader import load_processed_data

# Paths
MODEL_DIR = Path(__file__).parent.parent / "models" / "forecasting"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "forecasting"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def prepare_data():
    """Load and prepare time series data for forecasting"""
    print("Loading processed data...")
    df = load_processed_data()
    
    # Auto-detect datetime and value columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) == 0:
        # Try to convert common date column names
        for col in ['date', 'timestamp', 'time', 'datetime']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                datetime_cols = [col]
                break
    
    if len(datetime_cols) == 0:
        raise ValueError("No datetime column found. Please ensure data has a date/timestamp column.")
    
    datetime_col = datetime_cols[0]
    
    # Find numeric columns for forecasting
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for forecasting")
    
    # Use first numeric column as target, or look for common names
    target_col = None
    for col in ['count', 'traffic', 'visitors', 'movement', 'value']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        target_col = numeric_cols[0]
    
    # Create time series
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col)
    
    # Aggregate by datetime if needed
    ts_df = df.groupby(datetime_col)[target_col].sum().reset_index()
    ts_df.columns = ['ds', 'y']
    
    return ts_df

def train_arima(ts_df):
    """Train ARIMA model"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        print("\nTraining ARIMA...")
        
        # Split data
        train_size = int(len(ts_df) * 0.8)
        train = ts_df[:train_size]['y'].values
        test = ts_df[train_size:]['y'].values
        
        # Fit ARIMA model
        model = ARIMA(train, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=len(test))
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mae = mean_absolute_error(test, forecast)
        
        print(f"ARIMA RMSE: {rmse:.4f}")
        print(f"ARIMA MAE: {mae:.4f}")
        
        joblib.dump(fitted_model, MODEL_DIR / "arima.pkl")
        return fitted_model, rmse, mae
    except Exception as e:
        print(f"ARIMA training failed: {e}")
        return None, None, None

def train_prophet(ts_df):
    """Train Prophet model"""
    try:
        from prophet import Prophet
        print("\nTraining Prophet...")
        
        # Split data
        train_size = int(len(ts_df) * 0.8)
        train = ts_df[:train_size].copy()
        test = ts_df[train_size:].copy()
        
        # Fit Prophet model
        model = Prophet()
        model.fit(train)
        
        # Forecast
        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)
        
        # Get forecasted values for test period
        forecasted = forecast[-len(test):]['yhat'].values
        actual = test['y'].values
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual, forecasted))
        mae = mean_absolute_error(actual, forecasted)
        
        print(f"Prophet RMSE: {rmse:.4f}")
        print(f"Prophet MAE: {mae:.4f}")
        
        joblib.dump(model, MODEL_DIR / "prophet_model.pkl")
        return model, rmse, mae
    except Exception as e:
        print(f"Prophet training failed: {e}")
        return None, None, None

def main():
    """Main training function"""
    print("=" * 50)
    print("Training Forecasting Models")
    print("=" * 50)
    
    # Prepare data
    ts_df = prepare_data()
    print(f"\nTime series length: {len(ts_df)}")
    print(f"Date range: {ts_df['ds'].min()} to {ts_df['ds'].max()}")
    
    # Train models
    results = {}
    
    arima_model, arima_rmse, arima_mae = train_arima(ts_df)
    if arima_model is not None:
        results['arima'] = {
            'rmse': float(arima_rmse),
            'mae': float(arima_mae)
        }
    
    prophet_model, prophet_rmse, prophet_mae = train_prophet(ts_df)
    if prophet_model is not None:
        results['prophet'] = {
            'rmse': float(prophet_rmse),
            'mae': float(prophet_mae)
        }
    
    # Save results
    with open(RESULTS_DIR / "rmse.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()

