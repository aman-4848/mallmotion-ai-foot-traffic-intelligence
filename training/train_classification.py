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
    # Assuming zone prediction task - use SPACEID as target if available
    zone_cols = [col for col in df.columns if 'zone' in col.lower() or 'location' in col.lower() or 'spaceid' in col.lower()]
    target_col = zone_cols[0] if zone_cols else None
    
    # If no zone column, use a different approach - predict based on patterns
    if target_col is None:
        # Use last column or a suitable numeric column as target
        numeric_cols = [col for col in df.columns if df[col].dtype in [np.number]]
        target_col = numeric_cols[-1] if numeric_cols else df.columns[-1]
        print(f"Warning: No zone column found, using {target_col} as target")
    
    # Get all numeric columns as features, excluding target and ID columns
    # Use select_dtypes to get numeric columns (more reliable)
    numeric_df = df.select_dtypes(include=[np.number])
    all_numeric = list(numeric_df.columns)
    
    # Exclude target and ID columns
    exclude_list = [target_col]
    if 'USERID' in all_numeric:
        exclude_list.append('USERID')
    
    # Get feature columns (numeric, not in exclude list)
    feature_cols = [col for col in all_numeric if col not in exclude_list]
    
    # If still no features, use all numeric except target
    if len(feature_cols) == 0:
        print(f"Warning: No features after exclusion. Using all numeric except target.")
        feature_cols = [col for col in all_numeric if col != target_col]
    
    if len(feature_cols) == 0:
        raise ValueError(f"No numeric feature columns found. Total columns: {len(df.columns)}, Numeric: {len(all_numeric)}, Target: {target_col}, Exclude: {exclude_list}")
    
    print(f"Target column: {target_col}")
    print(f"Feature columns: {len(feature_cols)}")
    
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    # Handle categorical target - always encode to ensure consistent classes
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y = y_encoded
    
    # Save encoder
    preprocessing_dir = MODEL_DIR.parent / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, preprocessing_dir / "encoder.pkl")
    
    print(f"Target classes: {len(le.classes_)}")
    
    return X, y, feature_cols

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest classifier"""
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate ROC-AUC (handle binary and multi-class)
    try:
        if len(np.unique(y_test)) > 2:
            # Multi-class: use one-vs-rest
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr', average='macro')
        else:
            # Binary classification
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except Exception as e:
        print(f"Warning: Could not calculate ROC-AUC: {e}")
        roc_auc = np.nan
    
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
    # XGBoost requires all classes in training, so we need to handle this
    unique_classes = np.unique(np.concatenate([y_train, y_test]))
    model = XGBClassifier(random_state=42, n_jobs=-1)
    # Fit with all classes specified
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate ROC-AUC (handle binary and multi-class)
    try:
        if len(np.unique(y_test)) > 2:
            # Multi-class: use one-vs-rest
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr', average='macro')
        else:
            # Binary classification
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except Exception as e:
        print(f"Warning: Could not calculate ROC-AUC: {e}")
        roc_auc = np.nan
    
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

