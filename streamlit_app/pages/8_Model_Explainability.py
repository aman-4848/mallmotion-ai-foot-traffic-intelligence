"""
Model Explainability Page
Explain model decisions and feature importance
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from streamlit_app.utils.data_loader import load_processed_data
from features.feature_engineering import FeatureEngineer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

st.title("🧠 Model Explainability")
st.markdown("---")
models_dir = Path(__file__).parent.parent.parent / "models"
st.header("🔍 Feature Importance Analysis")
model_choice = st.selectbox("Select model:", ["XGBoost", "Random Forest", "Decision Tree", "SVM"])
# Map to file
model_files = {
    "XGBoost": "zone_xgb.pkl",
    "Random Forest": "zone_rf.pkl",
    "Decision Tree": "baseline_dt.pkl",
    "SVM": "zone_svm.pkl"
}
model_path = models_dir / "classification" / model_files[model_choice]
if model_path.exists():
    try:
        model = joblib.load(model_path)
        # Load data for feature names
        df = load_processed_data()
        fe = FeatureEngineer()
        df = fe.engineer_features(df)
        
        # Get feature columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SPACEID' in numeric_cols:
            numeric_cols.remove('SPACEID')
        if 'USERID' in numeric_cols:
            numeric_cols.remove('USERID')
        
        # Prepare data for importance calculation
        X = df[numeric_cols].fillna(0)
        
        # Get target for permutation importance
        target_col = 'SPACEID' if 'SPACEID' in df.columns else None
        if target_col is None:
            zone_cols = [col for col in df.columns if 'zone' in col.lower() or 'space' in col.lower()]
            target_col = zone_cols[0] if zone_cols else None
        
        if target_col is None:
            st.warning("Could not find target column for feature importance calculation.")
        else:
            y = df[target_col]
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Split data for permutation importance (use smaller subset for speed)
            X_sample = X.sample(min(5000, len(X)), random_state=42) if len(X) > 5000 else X
            y_sample = y_encoded[X_sample.index] if len(X) > 5000 else y_encoded
            
            # Get feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                # Tree-based models: use built-in feature importance
                n_features = len(model.feature_importances_)
                feature_names = numeric_cols[:n_features] if len(numeric_cols) >= n_features else numeric_cols
                importances = model.feature_importances_[:len(feature_names)]
                importance_type = "Built-in Feature Importance"
            elif model_choice == "SVM":
                # SVM: Use permutation importance
                with st.spinner("Calculating permutation importance for SVM (this may take a moment)..."):
                    perm_importance = permutation_importance(
                        model, X_sample.values, y_sample, 
                        n_repeats=5, random_state=42, n_jobs=-1, scoring='accuracy'
                    )
                    importances = perm_importance.importances_mean
                    feature_names = numeric_cols[:len(importances)] if len(importances) <= len(numeric_cols) else numeric_cols
                    importances = importances[:len(feature_names)]
                importance_type = "Permutation Importance"
            else:
                # For other models without feature_importances_, use permutation importance
                with st.spinner("Calculating permutation importance (this may take a moment)..."):
                    perm_importance = permutation_importance(
                        model, X_sample.values, y_sample, 
                        n_repeats=5, random_state=42, n_jobs=-1, scoring='accuracy'
                    )
                    importances = perm_importance.importances_mean
                    feature_names = numeric_cols[:len(importances)] if len(importances) <= len(numeric_cols) else numeric_cols
                    importances = importances[:len(feature_names)]
                importance_type = "Permutation Importance"
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Top features
            top_n = st.slider("Show top N features:", 10, 50, 20)
            top_features = importance_df.head(top_n)
            
            # Visualization
            color = 'purple' if model_choice == 'SVM' else 'steelblue'
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.barh(range(len(top_features)), top_features['Importance'].values, 
                   color=color, alpha=0.8, edgecolor='black')
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'].values)
            ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {top_n} Feature Importance - {model_choice} ({importance_type})', 
                       fontsize=14, fontweight='bold', pad=20)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Table
            st.subheader("Feature Importance Table")
            st.caption(f"*Using {importance_type}*")
            st.dataframe(top_features, use_container_width=True)
            
            # Insights
            if len(top_features) >= 5:
                st.subheader("💡 Insights")
                st.info(f"""
                **Top 5 Most Important Features:**
                1. {top_features.iloc[0]['Feature']} ({top_features.iloc[0]['Importance']:.4f})
                2. {top_features.iloc[1]['Feature']} ({top_features.iloc[1]['Importance']:.4f})
                3. {top_features.iloc[2]['Feature']} ({top_features.iloc[2]['Importance']:.4f})
                4. {top_features.iloc[3]['Feature']} ({top_features.iloc[3]['Importance']:.4f})
                5. {top_features.iloc[4]['Feature']} ({top_features.iloc[4]['Importance']:.4f})
                """)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
else:
    st.error("Model file not found.")
st.markdown("---")
# Model Comparison
st.header("📊 Model Comparison")
st.info("""
**Model Explainability:**

- **XGBoost**: Provides feature importance based on gain
- **Random Forest**: Uses mean decrease in impurity
- **Decision Tree**: Most interpretable, can visualize tree structure
- **SVM**: Uses permutation importance (measures how much model performance decreases when feature is shuffled)

**Feature Importance Methods:**
- **Built-in**: Tree-based models (XGBoost, Random Forest, Decision Tree) have native feature importance
- **Permutation Importance**: Used for SVM and other models without built-in importance

Feature importance helps understand which factors most influence zone predictions.
""")
