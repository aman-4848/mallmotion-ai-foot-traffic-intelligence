"""
Train Forecasting Models
Trains ARIMA and Random Forest Regressor forecasting models using processed data
Random Forest Regressor is more suitable for mall movement tracking with multiple features
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
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
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Try to convert common date column names
    if len(datetime_cols) == 0:
        for col in ['TIMESTAMP', 'timestamp', 'date', 'time', 'datetime', 'Date', 'Time']:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].dtype == 'datetime64[ns]':
                        datetime_cols = [col]
                        break
                except:
                    pass
    
    if len(datetime_cols) == 0:
        print("Warning: No datetime column found. Forecasting models may not work properly.")
        return None
    
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

def train_random_forest_forecast(ts_df):
    """Train Random Forest Regressor for forecasting - better for mall movement with multiple features"""
    print("\nTraining Random Forest Regressor for Forecasting...")
    
    try:
        # Create time-based features
        ts_df = ts_df.copy()
        ts_df['ds'] = pd.to_datetime(ts_df['ds'])
        ts_df['hour'] = ts_df['ds'].dt.hour
        ts_df['day_of_week'] = ts_df['ds'].dt.dayofweek
        ts_df['day_of_month'] = ts_df['ds'].dt.day
        ts_df['month'] = ts_df['ds'].dt.month
        ts_df['is_weekend'] = (ts_df['ds'].dt.dayofweek >= 5).astype(int)
        
        # Create lag features (previous values)
        ts_df['y_lag1'] = ts_df['y'].shift(1)
        ts_df['y_lag2'] = ts_df['y'].shift(2)
        ts_df['y_lag7'] = ts_df['y'].shift(7)  # Weekly pattern
        
        # Rolling window features
        ts_df['y_rolling_mean_7'] = ts_df['y'].rolling(window=7, min_periods=1).mean()
        ts_df['y_rolling_std_7'] = ts_df['y'].rolling(window=7, min_periods=1).std()
        
        # Fill NaN values from lag features
        ts_df = ts_df.bfill().fillna(0)
        
        # Prepare features
        feature_cols = ['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
                       'y_lag1', 'y_lag2', 'y_lag7', 'y_rolling_mean_7', 'y_rolling_std_7']
        
        X = ts_df[feature_cols].values
        y = ts_df['y'].values
        
        # Split data
        train_size = int(len(ts_df) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Regressor
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Forecast
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Random Forest Regressor RMSE: {rmse:.4f}")
        print(f"Random Forest Regressor MAE: {mae:.4f}")
        
        # Save model and scaler
        joblib.dump(model, MODEL_DIR / "rf_forecast.pkl")
        preprocessing_dir = MODEL_DIR.parent / "preprocessing"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, preprocessing_dir / "forecast_scaler.pkl")
        joblib.dump(feature_cols, preprocessing_dir / "forecast_features.pkl")
        
        return model, rmse, mae
    except Exception as e:
        print(f"Random Forest forecasting training failed: {e}")
        import traceback
        traceback.print_exc()
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
    
    rf_forecast_model, rf_rmse, rf_mae = train_random_forest_forecast(ts_df)
    if rf_forecast_model is not None:
        results['random_forest_regressor'] = {
            'rmse': float(rf_rmse),
            'mae': float(rf_mae)
        }
    
    # Save results
    with open(RESULTS_DIR / "rmse.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()

