"""
Data Explorer Page
Interactive data exploration and analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from streamlit_app.utils.data_loader import load_processed_data
from features.feature_engineering import FeatureEngineer

st.title("🔍 Data Explorer")
st.markdown("---")

# Load data
with st.spinner("Loading data..."):
    try:
        df_original = load_processed_data()
        fe = FeatureEngineer()
        df = fe.engineer_features(df_original)
        st.success("✅ Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Data Overview
st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Rows", f"{len(df):,}")
with col2:
    st.metric("Total Columns", len(df.columns))
with col3:
    st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
with col4:
    st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

st.markdown("---")

# Data Preview
st.header("👀 Data Preview")

preview_option = st.radio(
    "Select view:",
    ["First 10 rows", "Last 10 rows", "Random sample"],
    horizontal=True
)

if preview_option == "First 10 rows":
    st.dataframe(df.head(10), use_container_width=True)
elif preview_option == "Last 10 rows":
    st.dataframe(df.tail(10), use_container_width=True)
else:
    st.dataframe(df.sample(10), use_container_width=True)

st.markdown("---")

# Column Information
st.header("📋 Column Information")

col_type = st.selectbox("Filter by type:", ["All", "Numeric", "Categorical", "Datetime"])

if col_type == "All":
    display_cols = df.columns
elif col_type == "Numeric":
    display_cols = df.select_dtypes(include=[np.number]).columns
elif col_type == "Categorical":
    display_cols = df.select_dtypes(include=['object']).columns
else:
    display_cols = df.select_dtypes(include=['datetime64']).columns

col_info = pd.DataFrame({
    'Column': display_cols,
    'Type': [str(df[col].dtype) for col in display_cols],
    'Non-Null': [df[col].notna().sum() for col in display_cols],
    'Null': [df[col].isna().sum() for col in display_cols],
    'Unique': [df[col].nunique() for col in display_cols]
})

st.dataframe(col_info, use_container_width=True)

st.markdown("---")

# Statistical Summary
st.header("📈 Statistical Summary")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 0:
    selected_cols = st.multiselect(
        "Select columns for summary:",
        numeric_cols,
        default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
    )
    
    if selected_cols:
        st.dataframe(df[selected_cols].describe(), use_container_width=True)

st.markdown("---")

# Distribution Plots
st.header("📊 Distribution Analysis")

if len(numeric_cols) > 0:
    plot_col = st.selectbox("Select column to plot:", numeric_cols)
    
    if plot_col:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(df[plot_col].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_title(f'Distribution of {plot_col}')
        axes[0].set_xlabel(plot_col)
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(df[plot_col].dropna(), vert=True)
        axes[1].set_title(f'Box Plot: {plot_col}')
        axes[1].set_ylabel(plot_col)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

