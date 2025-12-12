"""
Overview Page - Dashboard Home
Shows project overview, key metrics, and quick access to all features
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from streamlit_app.utils.data_loader import load_processed_data, get_data_info

st.title("📊 Dashboard Overview")
st.markdown("---")

# Project Overview
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📁 Total Records", "15,839")
    st.caption("Processed data points")

with col2:
    st.metric("🔧 Features", "110")
    st.caption("Engineered features")

with col3:
    st.metric("🤖 Models Trained", "6")
    st.caption("ML models ready")

st.markdown("---")

# Key Metrics Section
st.header("🎯 Key Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

# Load results
try:
    results_dir = Path(__file__).parent.parent.parent / "results"
    
    # Classification metrics
    cls_results = None
    metrics_file = results_dir / "classification" / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            cls_results = json.load(f)
    
    # Clustering metrics
    clust_results = None
    clustering_file = results_dir / "clustering" / "silhouette_score.json"
    if clustering_file.exists():
        with open(clustering_file, 'r') as f:
            clust_results = json.load(f)
    
    if cls_results:
        with col1:
            best_acc = max(
                cls_results.get('random_forest', {}).get('accuracy', 0),
                cls_results.get('decision_tree', {}).get('accuracy', 0),
                cls_results.get('xgboost', {}).get('accuracy', 0),
                cls_results.get('logistic_regression', {}).get('accuracy', 0)
            )
            st.metric(
                "🏆 Best Classification",
                f"{best_acc*100:.2f}%",
                "Accuracy"
            )
    else:
        with col1:
            st.metric("🏆 Best Classification", "N/A", "Results not available")
    
    if clust_results:
        with col2:
            st.metric(
                "👥 Best Clustering",
                f"{clust_results['kmeans']['silhouette_score']:.3f}",
                "K-Means Silhouette"
            )
        
        with col3:
            st.metric(
                "📈 Clusters Found",
                f"{clust_results['kmeans']['n_clusters']}",
                "Customer Segments"
            )
    else:
        with col2:
            st.metric("👥 Best Clustering", "N/A", "Results not available")
        with col3:
            st.metric("📈 Clusters Found", "N/A", "Results not available")
    
    with col4:
        st.metric(
            "✅ Feature Engineering",
            "Complete",
            "30 new features"
        )
except Exception as e:
    st.warning(f"Could not load all metrics: {e}")
    # Show placeholder metrics
    with col1:
        st.metric("🏆 Best Classification", "N/A", "Loading...")
    with col2:
        st.metric("👥 Best Clustering", "N/A", "Loading...")
    with col3:
        st.metric("📈 Clusters Found", "N/A", "Loading...")

st.markdown("---")

# Quick Stats
st.header("📈 Quick Statistics")

try:
    df = load_processed_data()
    info = get_data_info(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Information")
        st.write(f"**Shape:** {info['shape'][0]:,} rows × {info['shape'][1]} columns")
        st.write(f"**Memory Usage:** {info['memory_usage'] / 1024**2:.2f} MB")
        st.write(f"**Missing Values:** {sum(info['null_counts'].values()):,}")
    
    with col2:
        st.subheader("🔧 Feature Engineering Status")
        st.success("✅ Feature Engineering Complete")
        st.write("• Missing values handled")
        st.write("• Temporal features extracted")
        st.write("• Domain features created")
        st.write("• All models trained")
except Exception as e:
    st.error(f"Error loading data: {e}")

st.markdown("---")

# Model Status
st.header("🤖 Model Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🎯 Classification")
    st.success("✅ 3 Models Trained")
    st.write("• Random Forest")
    st.write("• Decision Tree")
    st.write("• XGBoost (Best)")

with col2:
    st.subheader("👥 Clustering")
    st.success("✅ 2 Models Trained")
    st.write("• K-Means")
    st.write("• DBSCAN")

with col3:
    st.subheader("📈 Forecasting")
    st.info("✅ 1 Model Trained")
    st.write("• Random Forest Regressor")
    st.caption("ARIMA requires statsmodels")

st.markdown("---")

# Navigation Guide
st.header("🧭 Navigation Guide")

st.info("""
**Use the sidebar to navigate to different sections:**

- **📊 Overview** - Dashboard home (current page)
- **🔍 Data Explorer** - Explore and analyze the dataset
- **🗺️ Heatmaps** - Visualize movement patterns
- **🎯 Classification Results** - View classification model performance
- **👥 Clustering Insights** - Customer segmentation analysis
- **📈 Forecasting Traffic** - Traffic prediction models
- **🔮 Predict Next Zone** - Make predictions for new data
- **🧠 Model Explainability** - Understand model decisions
""")

st.markdown("---")

# Footer
st.caption("💡 **Tip:** All models are ready for predictions. Use 'Predict Next Zone' to make real-time predictions!")
