"""
Clustering Insights Page
Display clustering results and customer segmentation insights
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
from sklearn.preprocessing import StandardScaler

st.title("👥 Clustering Insights")
st.markdown("---")
# Load results
results_dir = Path(__file__).parent.parent.parent / "results"
models_dir = Path(__file__).parent.parent.parent / "models"

results = None
clustering_file = results_dir / "clustering" / "silhouette_score.json"
if clustering_file.exists():
    try:
        with open(clustering_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        st.error(f"Error loading results: {e}")
        results = None
else:
    st.warning("⚠️ Results file not found. Please run model training first.")
    st.info("💡 Run: `python training/train_clustering.py` to generate results.")
    results = {
        'kmeans': {'n_clusters': 0, 'silhouette_score': 0},
        'dbscan': {'n_clusters': 0, 'silhouette_score': 0, 'n_noise': 0}
    }

if results is None:
    st.stop()
# Clustering Metrics
st.header("📊 Clustering Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric(
        "K-Means 🏆",
        f"{results['kmeans']['silhouette_score']:.4f}",
        f"Silhouette Score ({results['kmeans']['n_clusters']} clusters)"
    )
with col2:
    st.metric(
        "DBSCAN",
        f"{results['dbscan'].get('silhouette_score', 0):.4f}",
        f"Silhouette Score ({results['dbscan']['n_clusters']} clusters, {results['dbscan']['n_noise']} noise)"
    )
st.markdown("---")
# Comparison Chart
st.header("📈 Model Comparison")
models = ['K-Means', 'DBSCAN']
scores = [
    results['kmeans']['silhouette_score'],
    results['dbscan'].get('silhouette_score', 0)
]
clusters = [
    results['kmeans']['n_clusters'],
    results['dbscan']['n_clusters']
]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Silhouette score comparison
axes[0].bar(models, scores, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
axes[0].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
axes[0].set_title('Clustering Models - Silhouette Score', fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 0.3])
axes[0].grid(True, alpha=0.3, axis='y')
for i, (model, score) in enumerate(zip(models, scores)):
    axes[0].text(i, score + 0.01, f'{score:.4f}', ha='center', fontweight='bold', fontsize=11)
    if score == max(scores):
        axes[0].text(i, score - 0.02, '🏆 BEST', ha='center', fontweight='bold', fontsize=10, color='red')
# Number of clusters
axes[1].bar(models, clusters, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
axes[1].set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
axes[1].set_title('Number of Clusters Found', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
for i, (model, n_clust) in enumerate(zip(models, clusters)):
    axes[1].text(i, n_clust + max(clusters) * 0.02, f'{n_clust}', ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
st.pyplot(fig)
plt.close()
st.markdown("---")
# Cluster Details
st.header("🔍 Cluster Details")
cluster_details = pd.DataFrame([
    {
        'Model': 'K-Means',
        'Clusters': results['kmeans']['n_clusters'],
        'Silhouette Score': results['kmeans']['silhouette_score'],
        'Noise Points': 0,
        'Type': 'Centroid-based'
    },
    {
        'Model': 'DBSCAN',
        'Clusters': results['dbscan']['n_clusters'],
        'Silhouette Score': results['dbscan'].get('silhouette_score', 0),
        'Noise Points': results['dbscan']['n_noise'],
        'Type': 'Density-based'
    }
])
st.dataframe(cluster_details, use_container_width=True)
st.markdown("---")
# Cluster Visualization (if model available)
st.header("📊 Cluster Visualization")
model_choice = st.selectbox("Select model to visualize:", ["K-Means", "DBSCAN"])
try:
    # Load data
    df = load_processed_data()
    fe = FeatureEngineer()
    df = fe.engineer_features(df)
    # Prepare features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'USERID' in numeric_cols:
        numeric_cols.remove('USERID')
    X = df[numeric_cols].fillna(0)
    # Scale
    scaler_path = models_dir / "preprocessing" / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    # Load model
    if model_choice == "K-Means":
        model_path = models_dir / "clustering" / "kmeans.pkl"
    else:
        model_path = models_dir / "clustering" / "dbscan.pkl"
    if model_path.exists():
        model = joblib.load(model_path)
        labels = model.predict(X_scaled) if hasattr(model, 'predict') else model.labels_
        # Cluster distribution
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['steelblue' if i != -1 else 'gray' for i in cluster_counts.index]
        ax.bar(range(len(cluster_counts)), cluster_counts.values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_choice} - Cluster Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(cluster_counts)))
        ax.set_xticklabels([f'C{i}' if i != -1 else 'Noise' for i in cluster_counts.index])
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        # Statistics
        st.subheader("Cluster Statistics")
        stats_df = pd.DataFrame({
            'Cluster': [f'C{i}' if i != -1 else 'Noise' for i in cluster_counts.index],
            'Count': cluster_counts.values,
            'Percentage': (cluster_counts.values / len(labels) * 100).round(2)
        })
        st.dataframe(stats_df, use_container_width=True)
except Exception as e:
    st.warning(f"Could not visualize clusters: {e}")
st.markdown("---")
# Insights
st.header("💡 Insights")
st.info("""
**Clustering Analysis:**
- **K-Means** found 5 distinct customer segments with good separation
- **DBSCAN** identified 396 micro-clusters, indicating high diversity in movement patterns
- K-Means is recommended for interpretable customer segmentation
- DBSCAN is useful for identifying outliers and noise points
""")