"""
Classification Results Page
Display classification model performance, metrics, and visualizations
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
from features.feature_engineering import FeatureEngineer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

st.title("🎯 Classification Results")
st.markdown("---")
# Load results
results_dir = Path(__file__).parent.parent.parent / "results"
models_dir = Path(__file__).parent.parent.parent / "models"

results = None
metrics_file = results_dir / "classification" / "metrics.json"
if metrics_file.exists():
    try:
        with open(metrics_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        st.error(f"Error loading results: {e}")
        results = None
else:
    st.warning("⚠️ Results file not found. Please run model training first.")
    st.info("💡 Run: `python training/train_classification.py` to generate results.")
    results = {
        'random_forest': {'accuracy': 0},
        'decision_tree': {'accuracy': 0},
        'xgboost': {'accuracy': 0},
        'logistic_regression': {'accuracy': 0}
    }

if results is None:
    st.stop()
# Model Performance Metrics
st.header("📊 Model Performance Metrics")

# Find best model
all_accuracies = {
    'random_forest': results.get('random_forest', {}).get('accuracy', 0),
    'decision_tree': results.get('decision_tree', {}).get('accuracy', 0),
    'xgboost': results.get('xgboost', {}).get('accuracy', 0),
    'logistic_regression': results.get('logistic_regression', {}).get('accuracy', 0)
}
best_model = max(all_accuracies.items(), key=lambda x: x[1])[0]

col1, col2, col3, col4 = st.columns(4)
with col1:
    badge = "🏆" if best_model == 'random_forest' else ""
    st.metric(
        f"Random Forest {badge}",
        f"{results.get('random_forest', {}).get('accuracy', 0)*100:.2f}%",
        "Accuracy"
    )
    if 'roc_auc' in results.get('random_forest', {}) and not pd.isna(results['random_forest'].get('roc_auc')):
        st.caption(f"ROC-AUC: {results['random_forest']['roc_auc']:.4f}")
with col2:
    badge = "🏆" if best_model == 'decision_tree' else ""
    st.metric(
        f"Decision Tree {badge}",
        f"{results.get('decision_tree', {}).get('accuracy', 0)*100:.2f}%",
        "Accuracy"
    )
with col3:
    badge = "🏆" if best_model == 'xgboost' else ""
    st.metric(
        f"XGBoost {badge}",
        f"{results.get('xgboost', {}).get('accuracy', 0)*100:.2f}%",
        "Best Model" if best_model == 'xgboost' else ""
    )
    if 'roc_auc' in results.get('xgboost', {}) and not pd.isna(results['xgboost'].get('roc_auc')):
        st.caption(f"ROC-AUC: {results['xgboost']['roc_auc']:.4f}")
with col4:
    badge = "🏆" if best_model == 'logistic_regression' else ""
    st.metric(
        f"Logistic Regression {badge}",
        f"{results.get('logistic_regression', {}).get('accuracy', 0)*100:.2f}%",
        "Best Model" if best_model == 'logistic_regression' else ""
    )
    if 'roc_auc' in results.get('logistic_regression', {}) and not pd.isna(results['logistic_regression'].get('roc_auc')):
        st.caption(f"ROC-AUC: {results['logistic_regression']['roc_auc']:.4f}")
st.markdown("---")
# Comparison Chart
st.header("📈 Model Comparison")
models = []
accuracies = []
colors = []

if 'random_forest' in results:
    models.append('Random Forest')
    accuracies.append(results['random_forest']['accuracy'])
    colors.append('steelblue')
if 'decision_tree' in results:
    models.append('Decision Tree')
    accuracies.append(results['decision_tree']['accuracy'])
    colors.append('coral')
if 'xgboost' in results:
    models.append('XGBoost')
    accuracies.append(results['xgboost']['accuracy'])
    colors.append('lightgreen')
if 'logistic_regression' in results:
    models.append('Logistic Regression')
    accuracies.append(results['logistic_regression']['accuracy'])
    colors.append('purple')

if models and accuracies:
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Classification Models - Accuracy Comparison', fontsize=14, fontweight='bold')
    if accuracies:
        min_acc = min(accuracies)
        max_acc = max(accuracies)
        ax.set_ylim([max(0.0, min_acc - 0.05), min(1.0, max_acc + 0.05)])
    ax.grid(True, alpha=0.3, axis='y')
    for i, (model, acc) in enumerate(zip(models, accuracies)):
        ax.text(i, acc + (max(accuracies) - min(accuracies)) * 0.02 if max(accuracies) > min(accuracies) else 0.02, f'{acc:.4f}', 
               ha='center', fontweight='bold', fontsize=11)
        if acc == max(accuracies):
            ax.text(i, acc - (max(accuracies) - min(accuracies)) * 0.05 if max(accuracies) > min(accuracies) else 0.05, '🏆 BEST', 
                   ha='center', fontweight='bold', fontsize=10, color='red')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
st.markdown("---")
# Detailed Results Table
st.header("📋 Detailed Results")
results_list = []

if 'random_forest' in results:
    results_list.append({
        'Model': 'Random Forest',
        'Accuracy': results['random_forest']['accuracy'],
        'ROC-AUC': results['random_forest'].get('roc_auc', 'N/A'),
        'Type': 'Ensemble'
    })
if 'decision_tree' in results:
    results_list.append({
        'Model': 'Decision Tree',
        'Accuracy': results['decision_tree']['accuracy'],
        'ROC-AUC': 'N/A',
        'Type': 'Baseline'
    })
if 'xgboost' in results:
    results_list.append({
        'Model': 'XGBoost',
        'Accuracy': results['xgboost']['accuracy'],
        'ROC-AUC': results['xgboost'].get('roc_auc', 'N/A'),
        'Type': 'Gradient Boosting'
    })
if 'logistic_regression' in results:
    results_list.append({
        'Model': 'Logistic Regression',
        'Accuracy': results['logistic_regression']['accuracy'],
        'ROC-AUC': results['logistic_regression'].get('roc_auc', 'N/A'),
        'Type': 'Logistic Regression'
    })

if results_list:
    results_df = pd.DataFrame(results_list)
    # Sort by accuracy descending
    results_df = results_df.sort_values('Accuracy', ascending=False)
    # Mark best model
    results_df['Best'] = results_df['Accuracy'] == results_df['Accuracy'].max()
    results_df['Best'] = results_df['Best'].map({True: '🏆', False: ''})
    st.dataframe(results_df, use_container_width=True)
st.markdown("---")
# Feature Importance (if model available)
st.header("🔍 Feature Importance")
model_choice = st.selectbox("Select model for feature importance:", ["XGBoost", "Random Forest", "Logistic Regression"])
try:
    if model_choice == "XGBoost":
        model_path = models_dir / "classification" / "zone_xgb.pkl"
    elif model_choice == "Random Forest":
        model_path = models_dir / "classification" / "zone_rf.pkl"
    else:  # Logistic Regression
        model_path = models_dir / "classification" / "zone_lr.pkl"
    
    if model_path.exists():
        model = joblib.load(model_path)
        
        # Load data to get feature names
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
            from sklearn.preprocessing import LabelEncoder
            y = df[target_col]
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Split data for permutation importance (use smaller subset for speed)
            X_sample = X.sample(min(5000, len(X)), random_state=42) if len(X) > 5000 else X
            y_sample = y_encoded[X_sample.index] if len(X) > 5000 else y_encoded
            
            # Get feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                # Tree-based models: use built-in feature importance
                if len(model.feature_importances_) <= len(numeric_cols):
                    feature_names = numeric_cols[:len(model.feature_importances_)]
                    importances = model.feature_importances_
                else:
                    feature_names = numeric_cols
                    importances = model.feature_importances_[:len(numeric_cols)]
                
                importance_type = "Built-in Feature Importance"
            elif model_choice == "Logistic Regression":
                # Logistic Regression: Use permutation importance
                with st.spinner("Calculating permutation importance for Logistic Regression (this may take a moment)..."):
                    # Use a smaller sample for permutation importance to speed it up
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
            }).sort_values('Importance', ascending=False).head(20)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            color = 'purple' if model_choice == 'Logistic Regression' else 'steelblue'
            ax.barh(range(len(importance_df)), importance_df['Importance'].values, 
                   color=color, alpha=0.8, edgecolor='black')
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['Feature'].values)
            ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
            ax.set_title(f'Top 20 Feature Importance - {model_choice} ({importance_type})', 
                        fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Table
            st.caption(f"*Using {importance_type}*")
            st.dataframe(importance_df, use_container_width=True)
except Exception as e:
    st.warning(f"Could not load feature importance: {e}")
st.markdown("---")
# Model Information
st.header("ℹ️ Model Information")
st.info("""
**Classification Models:**

- **Random Forest**: Ensemble method using multiple decision trees
- **Decision Tree**: Baseline model, easy to interpret
- **XGBoost**: Gradient boosting algorithm, high performance
- **Logistic Regression**: Linear model, fast and interpretable

**Target:** Predicting next zone/location (SPACEID)
**Best Model:** Check metrics above to see current best performer
""")
