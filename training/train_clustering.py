"""
Train Clustering Models
Trains K-Means and DBSCAN clustering models using processed data
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
import json
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from streamlit_app.utils.data_loader import load_processed_data
from features.feature_engineering import FeatureEngineer

# Paths
MODEL_DIR = Path(__file__).parent.parent / "models" / "clustering"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "clustering"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def prepare_data(n_clusters=5):
    """Load and prepare data for clustering"""
    print("Loading processed data...")
    df = load_processed_data()
    
    print("Engineering features...")
    # Use comprehensive feature engineering
    fe = FeatureEngineer()
    df = fe.engineer_features(df)
    
    # Select numeric features only (use select_dtypes for reliability)
    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = list(numeric_df.columns)
    
    # Exclude ID columns
    if 'USERID' in feature_cols:
        feature_cols.remove('USERID')
    
    if len(feature_cols) == 0:
        raise ValueError(f"No numeric feature columns found. Total columns: {len(df.columns)}, Numeric: {len(numeric_df.columns)}")
    
    X = df[feature_cols].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    scaler_dir = MODEL_DIR.parent / "preprocessing"
    scaler_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_dir / "scaler.pkl")
    
    return X_scaled, feature_cols, n_clusters

def train_kmeans(X, n_clusters):
    """Train K-Means clustering"""
    print(f"\nTraining K-Means with {n_clusters} clusters...")
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    
    silhouette = silhouette_score(X, labels)
    print(f"K-Means Silhouette Score: {silhouette:.4f}")
    
    joblib.dump(model, MODEL_DIR / "kmeans.pkl")
    return model, labels, silhouette

def train_dbscan(X):
    """Train DBSCAN clustering"""
    print("\nTraining DBSCAN...")
    model = DBSCAN(eps=0.5, min_samples=5)
    labels = model.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"DBSCAN found {n_clusters} clusters")
    print(f"Number of noise points: {n_noise}")
    
    if n_clusters > 1:
        silhouette = silhouette_score(X, labels)
        print(f"DBSCAN Silhouette Score: {silhouette:.4f}")
    else:
        silhouette = -1
        print("DBSCAN Silhouette Score: N/A (too few clusters)")
    
    joblib.dump(model, MODEL_DIR / "dbscan.pkl")
    return model, labels, silhouette

def main():
    """Main training function"""
    print("=" * 50)
    print("Training Clustering Models")
    print("=" * 50)
    
    # Prepare data
    n_clusters = 5  # Can be adjusted
    X, feature_cols, n_clusters = prepare_data(n_clusters)
    
    print(f"\nData shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Train models
    results = {}
    
    kmeans_model, kmeans_labels, kmeans_sil = train_kmeans(X, n_clusters)
    results['kmeans'] = {
        'n_clusters': int(n_clusters),
        'silhouette_score': float(kmeans_sil)
    }
    
    dbscan_model, dbscan_labels, dbscan_sil = train_dbscan(X)
    results['dbscan'] = {
        'n_clusters': int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)),
        'silhouette_score': float(dbscan_sil) if dbscan_sil != -1 else None,
        'n_noise': int(list(dbscan_labels).count(-1))
    }
    
    # Save results
    with open(RESULTS_DIR / "silhouette_score.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()

