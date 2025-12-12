"""
Forecasting Traffic Page
Display forecasting model results and predictions
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from streamlit_app.utils.data_loader import load_processed_data

st.title("📈 Forecasting Traffic")
st.markdown("---")
# Load results
results_dir = Path(__file__).parent.parent.parent / "results"
models_dir = Path(__file__).parent.parent.parent / "models"
try:
    with open(results_dir / "forecasting" / "rmse.json", 'r') as f:
        results = json.load(f)
except Exception as e:
    st.warning("Forecasting results not available. Models may need to be trained.")
    results = {}
# Forecasting Metrics
st.header("📊 Forecasting Performance")
if results:
    cols = st.columns(len(results))
    for idx, (model_name, metrics) in enumerate(results.items()):
        with cols[idx]:
            st.metric(
                model_name.upper(),
                f"{metrics['rmse']:.2e}",
                f"RMSE"
            )
            st.caption(f"MAE: {metrics['mae']:.2e}")
else:
    st.info("Forecasting models are being prepared. Check back soon!")
st.markdown("---")
# Model Comparison
if results:
    st.header("📈 Model Comparison")
    models = list(results.keys())
    rmse_values = [results[m]['rmse'] for m in models]
    mae_values = [results[m]['mae'] for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # RMSE comparison
    axes[0].bar(models, rmse_values, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
    axes[0].set_title('Forecasting Models - RMSE Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, (model, rmse) in enumerate(zip(models, rmse_values)):
        axes[0].text(i, rmse + max(rmse_values) * 0.02, f'{rmse:.2e}', 
                    ha='center', fontweight='bold', fontsize=10)
    # MAE comparison
    axes[1].bar(models, mae_values, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('MAE (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1].set_title('Forecasting Models - MAE Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, (model, mae) in enumerate(zip(models, mae_values)):
        axes[1].text(i, mae + max(mae_values) * 0.02, f'{mae:.2e}', 
                    ha='center', fontweight='bold', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown("---")
# Forecasting Details
st.header("📋 Forecasting Details")
if results:
    forecast_df = pd.DataFrame([
        {
            'Model': model.upper(),
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae']
        }
        for model, metrics in results.items()
    ])
    st.dataframe(forecast_df, use_container_width=True)
else:
    st.info("No forecasting results available yet.")
st.markdown("---")
# Model Information
st.header("ℹ️ Forecasting Models")
st.info("""
**Forecasting Models:**
- **ARIMA**: Statistical time series model (requires statsmodels)
- **Random Forest Regressor**: Ensemble method, handles multiple features and non-linear patterns well
**Use Case:** Predict future traffic patterns and customer movement trends
""")