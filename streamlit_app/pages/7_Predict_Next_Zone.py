"""
Predict Next Zone Page
Interactive prediction interface for making real-time predictions
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from streamlit_app.utils.data_loader import load_processed_data
from features.feature_engineering import FeatureEngineer

st.title("🔮 Predict Next Zone")
st.markdown("---")
models_dir = Path(__file__).parent.parent.parent / "models"
# Model Selection
st.header("🤖 Select Model")
model_choice = st.selectbox(
    "Choose classification model:",
    ["XGBoost (Best)", "Decision Tree", "Logistic Regression"]
    # Note: Random Forest (402MB) excluded due to GitHub file size limits
)
# Map selection to file
model_files = {
    "XGBoost (Best)": "zone_xgb.pkl",
    "Random Forest": "zone_rf.pkl",
    "Decision Tree": "baseline_dt.pkl",
    "Logistic Regression": "zone_lr.pkl"
}
model_path = models_dir / "classification" / model_files[model_choice]
if not model_path.exists():
    st.error(f"⚠️ Model file not found: {model_path}")
    st.warning(f"**Expected location:** `{model_path}`")
    st.info("""
    **To fix this issue:**
    1. Ensure model files are trained and saved in the `models/` directory
    2. Run: `python training/train_classification.py` to generate models
    3. Check that the model file exists at the expected path
    
    **Note:** If deploying, ensure model files are committed to git repository.
    """)
    
    # Show available models
    st.subheader("📋 Available Models")
    classification_dir = models_dir / "classification"
    if classification_dir.exists():
        available = [f.name for f in classification_dir.glob("*.pkl")]
        if available:
            st.success(f"✅ Found {len(available)} model file(s) in classification directory")
            st.code("\n".join(available))
        else:
            st.warning("❌ No model files found in classification directory")
    else:
        st.warning("❌ Classification models directory does not exist")
    st.stop()
# Load model and scaler (if needed for Logistic Regression)
try:
    model = joblib.load(model_path)
    
    # Load scaler for Logistic Regression
    scaler = None
    if 'Logistic Regression' in model_choice:
        scaler_path = models_dir.parent / "preprocessing" / "lr_scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            st.success(f"✅ {model_choice} model and scaler loaded successfully!")
        else:
            st.warning(f"⚠️ Scaler not found for Logistic Regression. Predictions may be inaccurate.")
            st.success(f"✅ {model_choice} model loaded successfully!")
    else:
        st.success(f"✅ {model_choice} model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
st.markdown("---")
# Prediction Mode
st.header("📝 Prediction Mode")
prediction_mode = st.radio(
    "Choose prediction mode:",
    ["Single Prediction", "Batch Prediction"],
    horizontal=True
)
if prediction_mode == "Single Prediction":
    # Single prediction form
    st.subheader("Enter Features for Prediction")
    # Load data to get feature ranges
    try:
        df = load_processed_data()
        fe = FeatureEngineer()
        df = fe.engineer_features(df)
        # Get numeric features (excluding target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SPACEID' in numeric_cols:
            numeric_cols.remove('SPACEID')
        if 'USERID' in numeric_cols:
            numeric_cols.remove('USERID')
        # Create input form
        input_features = {}
        cols = st.columns(3)
        # Show first 9 features as example
        sample_features = numeric_cols[:9] if len(numeric_cols) >= 9 else numeric_cols
        for idx, feature in enumerate(sample_features):
            with cols[idx % 3]:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                input_features[feature] = st.number_input(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100
                )
        # Predict button
        if st.button("🔮 Predict Next Zone", type="primary"):
            try:
                # Prepare input - get all feature values
                feature_values = [input_features.get(f, df[f].mean()) for f in numeric_cols]
                
                # For tree-based models, check feature count
                if hasattr(model, 'feature_importances_'):
                    if len(feature_values) != len(model.feature_importances_):
                        feature_values = feature_values[:len(model.feature_importances_)]
                
                X_input = np.array([feature_values])
                
                # Scale features if Logistic Regression
                if scaler is not None:
                    X_input = scaler.transform(X_input)
                
                # Make prediction
                prediction = model.predict(X_input)[0]
                probabilities = model.predict_proba(X_input)[0] if hasattr(model, 'predict_proba') else None
                # Display result
                st.success(f"✅ Predicted Zone: **{prediction}**")
                if probabilities is not None:
                    st.subheader("Prediction Probabilities (Top 5)")
                    top_indices = np.argsort(probabilities)[-5:][::-1]
                    prob_df = pd.DataFrame({
                        'Zone': top_indices,
                        'Probability': probabilities[top_indices]
                    })
                    st.dataframe(prob_df, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")
    except Exception as e:
        st.error(f"Error loading data: {e}")
else:
    # Batch prediction
    st.subheader("Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df_upload)} records")
            # Process and predict
            if st.button("🔮 Predict All", type="primary"):
                with st.spinner("Processing predictions..."):
                    # Apply feature engineering
                    fe = FeatureEngineer()
                    df_processed = fe.engineer_features(df_upload)
                    # Get features
                    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
                    if 'SPACEID' in numeric_cols:
                        numeric_cols.remove('SPACEID')
                    if 'USERID' in numeric_cols:
                        numeric_cols.remove('USERID')
                    X = df_processed[numeric_cols].fillna(0)
                    
                    # Scale features if Logistic Regression
                    if scaler is not None:
                        X = scaler.transform(X)
                    
                    # Predict
                    predictions = model.predict(X)
                    # Add predictions to dataframe
                    df_upload['Predicted_Zone'] = predictions
                    st.success(f"✅ Predictions complete for {len(predictions)} records!")
                    st.dataframe(df_upload, use_container_width=True)
                    # Download button
                    csv = df_upload.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")
st.markdown("---")
# Model Info
st.header("ℹ️ Model Information")
# Determine model type
if 'XGBoost' in model_choice:
    model_type = 'Gradient Boosting'
elif 'Random' in model_choice:
    model_type = 'Ensemble'
elif 'Logistic Regression' in model_choice:
    model_type = 'Logistic Regression'
else:
    model_type = 'Decision Tree'

st.info(f"""
**Using:** {model_choice}
**Model Type:** {model_type}
**Best Performance:** Check Classification Results page for current best model
""")