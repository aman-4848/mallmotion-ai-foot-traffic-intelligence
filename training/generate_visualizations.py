"""
Generate All Visualizations
Creates and saves all necessary visualizations for Streamlit dashboard
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from pathlib import Path
import sys
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))
from streamlit_app.utils.data_loader import load_processed_data
from features.feature_engineering import FeatureEngineer
from sklearn.preprocessing import LabelEncoder

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS_DIR = Path(__file__).parent.parent / "models"

def generate_classification_visualizations():
    """Generate classification model visualizations"""
    print("Generating classification visualizations...")
    
    # Load data
    df = load_processed_data()
    fe = FeatureEngineer()
    df = fe.engineer_features(df)
    
    # Prepare data
    target_col = 'SPACEID'
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if 'USERID' in numeric_cols:
        numeric_cols.remove('USERID')
    
    X = df[numeric_cols].fillna(0)
    y = df[target_col]
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load best model (XGBoost)
    model_path = MODELS_DIR / "classification" / "zone_xgb.pkl"
    if model_path.exists():
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix - XGBoost', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "classification" / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        print("  ✓ Saved: confusion_matrix.png")
        plt.close()
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': numeric_cols[:len(model.feature_importances_)],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(importance_df)), importance_df['importance'].values, color='steelblue')
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'].values)
            ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
            ax.set_title('Top 20 Feature Importance - XGBoost', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / "classification" / "feature_importance.png", dpi=300, bbox_inches='tight')
            print("  ✓ Saved: feature_importance.png")
            plt.close()

def generate_clustering_visualizations():
    """Generate clustering visualizations"""
    print("Generating clustering visualizations...")
    
    # Load data
    df = load_processed_data()
    fe = FeatureEngineer()
    df = fe.engineer_features(df)
    
    # Prepare data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'USERID' in numeric_cols:
        numeric_cols.remove('USERID')
    X = df[numeric_cols].fillna(0)
    
    # Load scaler and model
    scaler_path = MODELS_DIR / "preprocessing" / "scaler.pkl"
    model_path = MODELS_DIR / "clustering" / "kmeans.pkl"
    
    if scaler_path.exists() and model_path.exists():
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        
        X_scaled = scaler.transform(X)
        labels = model.predict(X_scaled)
        
        # Cluster distribution
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(cluster_counts)), cluster_counts.values, color='steelblue', alpha=0.8)
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
        ax.set_title('K-Means Cluster Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(cluster_counts)))
        ax.set_xticklabels([f'C{i}' for i in cluster_counts.index])
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "clustering" / "cluster_plot.png", dpi=300, bbox_inches='tight')
        print("  ✓ Saved: cluster_plot.png")
        plt.close()

def generate_forecasting_visualizations():
    """Generate forecasting visualizations"""
    print("Generating forecasting visualizations...")
    
    # Load data
    df = load_processed_data()
    
    # Try to create time series
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) == 0:
        for col in ['TIMESTAMP', 'timestamp', 'date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                datetime_cols = [col]
                break
    
    if len(datetime_cols) > 0:
        datetime_col = datetime_cols[0]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_col = numeric_cols[0] if len(numeric_cols) > 0 else None
        
        if value_col:
            ts_df = df.groupby(df[datetime_col].dt.date)[value_col].sum().reset_index()
            ts_df.columns = ['ds', 'y']
            ts_df['ds'] = pd.to_datetime(ts_df['ds'])
            
            # Simple forecast plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(ts_df['ds'], ts_df['y'], linewidth=2, color='steelblue', label='Actual')
            ax.set_title('Time Series Data', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / "forecasting" / "forecast_plot.png", dpi=300, bbox_inches='tight')
            print("  ✓ Saved: forecast_plot.png")
            plt.close()

def main():
    """Generate all visualizations"""
    print("=" * 60)
    print("GENERATING ALL VISUALIZATIONS")
    print("=" * 60)
    
    # Create directories
    (RESULTS_DIR / "classification").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "clustering").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "forecasting").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "comparisons").mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    generate_classification_visualizations()
    generate_clustering_visualizations()
    generate_forecasting_visualizations()
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS GENERATED!")
    print("=" * 60)

if __name__ == "__main__":
    main()

