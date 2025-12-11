"""
Train Classification Models
Trains zone prediction classification models using processed data
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from streamlit_app.utils.data_loader import load_processed_data
from features.feature_engineering import FeatureEngineer

# Paths
MODEL_DIR = Path(__file__).parent.parent / "models" / "classification"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "classification"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def prepare_data():
    """Load and prepare data for classification"""
    print("Loading processed data...")
    df = load_processed_data()
    
    print("Engineering features...")
    # Use comprehensive feature engineering
    fe = FeatureEngineer()
    df = fe.engineer_features(df)
    
    # Auto-detect target and feature columns
    # Assuming zone prediction task
    zone_cols = [col for col in df.columns if 'zone' in col.lower() or 'location' in col.lower()]
    target_col = zone_cols[0] if zone_cols else df.columns[-1]
    
    # Exclude non-feature columns
    exclude_cols = ['target', target_col] + [col for col in df.columns if df[col].dtype == 'object']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.number]]
    
    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found")
    
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    # Handle categorical target
    if y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        joblib.dump(le, MODEL_DIR.parent / "preprocessing" / "encoder.pkl")
    
    return X, y, feature_cols

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest classifier"""
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr') if len(np.unique(y_test)) > 2 else roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print(f"Random Forest ROC-AUC: {roc_auc:.4f}")
    
    joblib.dump(model, MODEL_DIR / "zone_rf.pkl")
    return model, accuracy, roc_auc

def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train Decision Tree classifier"""
    print("\nTraining Decision Tree...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
    
    joblib.dump(model, MODEL_DIR / "baseline_dt.pkl")
    return model, accuracy

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier"""
    print("\nTraining XGBoost...")
    model = XGBClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr') if len(np.unique(y_test)) > 2 else roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print(f"XGBoost Accuracy: {accuracy:.4f}")
    print(f"XGBoost ROC-AUC: {roc_auc:.4f}")
    
    joblib.dump(model, MODEL_DIR / "zone_xgb.pkl")
    return model, accuracy, roc_auc

def main():
    """Main training function"""
    print("=" * 50)
    print("Training Classification Models")
    print("=" * 50)
    
    # Prepare data
    X, y, feature_cols = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Train models
    results = {}
    
    rf_model, rf_acc, rf_auc = train_random_forest(X_train, y_train, X_test, y_test)
    results['random_forest'] = {'accuracy': float(rf_acc), 'roc_auc': float(rf_auc)}
    
    dt_model, dt_acc = train_decision_tree(X_train, y_train, X_test, y_test)
    results['decision_tree'] = {'accuracy': float(dt_acc)}
    
    xgb_model, xgb_acc, xgb_auc = train_xgboost(X_train, y_train, X_test, y_test)
    results['xgboost'] = {'accuracy': float(xgb_acc), 'roc_auc': float(xgb_auc)}
    
    # Save results
    with open(RESULTS_DIR / "metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"\nBest model: {max(results.items(), key=lambda x: x[1].get('accuracy', 0))[0]}")

if __name__ == "__main__":
    main()

