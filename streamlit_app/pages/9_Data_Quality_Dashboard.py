"""
============================================================================
DATA QUALITY DASHBOARD
============================================================================
Comprehensive data quality monitoring and analysis dashboard.
Provides insights into data completeness, validity, consistency, and quality metrics.
============================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import logging
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import project utilities
from streamlit_app.utils.data_loader import load_processed_data, get_data_info
from streamlit_app.utils.error_handler import handle_errors, safe_execute, show_error
from streamlit_app.utils.charts import setup_style, create_bar_chart, create_line_plot, create_heatmap
from streamlit_app.utils.validation import validate_dataframe
from features.feature_engineering import FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
PAGE_TITLE = "🔍 Data Quality Dashboard"
PAGE_DESCRIPTION = """
Comprehensive data quality monitoring and analysis dashboard. 
Monitor data completeness, validity, consistency, and identify quality issues.
"""
PAGE_ICON = "🔍"

# ============================================================================
# CONSTANTS
# ============================================================================
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MONITORING_DIR = Path(__file__).parent.parent.parent / "monitoring"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=True)
@handle_errors
def load_data() -> Optional[pd.DataFrame]:
    """
    Load and process data with caching and error handling.
    
    Returns:
        Processed DataFrame or None if loading fails
    """
    try:
        logger.info("Loading processed data for quality analysis...")
        df = load_processed_data()
        
        if df is None or df.empty:
            raise ValueError("Data is empty or None")
        
        logger.info(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        st.error(f"❌ Data file not found. Please ensure data files are in the correct location.")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        show_error(e, context="Data Loading")
        return None


@handle_errors
def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive data quality metrics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    if df is None or df.empty:
        return {}
    
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    metrics = {
        'total_records': len(df),
        'total_features': len(df.columns),
        'total_cells': total_cells,
        'missing_values': missing_cells,
        'missing_percentage': (missing_cells / total_cells * 100) if total_cells > 0 else 0,
        'completeness': (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 100,
        'duplicate_rows': duplicate_rows,
        'duplicate_percentage': (duplicate_rows / len(df) * 100) if len(df) > 0 else 0,
        'uniqueness': (1 - duplicate_rows / len(df)) * 100 if len(df) > 0 else 100,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Column-level metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    metrics['numeric_features'] = len(numeric_cols)
    metrics['categorical_features'] = len(categorical_cols)
    
    # Missing values by column
    missing_by_col = df.isnull().sum()
    metrics['columns_with_missing'] = (missing_by_col > 0).sum()
    metrics['columns_complete'] = (missing_by_col == 0).sum()
    
    # Data types distribution
    metrics['data_types'] = df.dtypes.value_counts().to_dict()
    
    return metrics


@handle_errors
def get_column_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate quality report for each column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with quality metrics per column
    """
    quality_data = []
    
    for col in df.columns:
        col_data = df[col]
        missing_count = col_data.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        quality_data.append({
            'Column': col,
            'Data Type': str(col_data.dtype),
            'Missing Count': missing_count,
            'Missing %': round(missing_pct, 2),
            'Completeness %': round(100 - missing_pct, 2),
            'Unique Values': col_data.nunique(),
            'Duplicate Values': len(df) - col_data.nunique(),
        })
        
        # Add statistics for numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            quality_data[-1].update({
                'Mean': round(col_data.mean(), 2) if not pd.isna(col_data.mean()) else None,
                'Std Dev': round(col_data.std(), 2) if not pd.isna(col_data.std()) else None,
                'Min': round(col_data.min(), 2) if not pd.isna(col_data.min()) else None,
                'Max': round(col_data.max(), 2) if not pd.isna(col_data.max()) else None,
            })
    
    return pd.DataFrame(quality_data)


@handle_errors
def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """
    Detect outliers in a numeric column.
    
    Args:
        df: Input DataFrame
        column: Column name to analyze
        method: Method for outlier detection ('iqr' or 'zscore')
        
    Returns:
        DataFrame with outlier information
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return pd.DataFrame()
    
    col_data = df[column].dropna()
    
    if method == 'iqr':
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    else:  # zscore
        z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
        outlier_indices = z_scores[z_scores > 3].index
        outliers = df.loc[outlier_indices]
    
    return outliers


# ============================================================================
# PAGE LAYOUT
# ============================================================================

def main():
    """Main page function"""
    
    # Page Header
    st.title(PAGE_TITLE)
    st.markdown(PAGE_DESCRIPTION)
    st.markdown("---")
    
    # ========================================================================
    # SIDEBAR CONFIGURATION
    # ========================================================================
    with st.sidebar:
        st.header("⚙️ Dashboard Settings")
        
        # Data refresh option
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Quality thresholds
        st.subheader("📊 Quality Thresholds")
        completeness_threshold = st.slider(
            "Completeness Threshold (%)",
            0, 100, 95,
            help="Minimum acceptable completeness percentage"
        )
        uniqueness_threshold = st.slider(
            "Uniqueness Threshold (%)",
            0, 100, 90,
            help="Minimum acceptable uniqueness percentage"
        )
        
        st.markdown("---")
        
        # Display options
        st.subheader("📈 Display Options")
        show_detailed_report = st.checkbox("Show Detailed Column Report", value=True)
        show_outlier_analysis = st.checkbox("Show Outlier Analysis", value=False)
        show_data_types = st.checkbox("Show Data Type Distribution", value=True)
        
        st.markdown("---")
        
        # Information section
        with st.expander("ℹ️ About This Dashboard"):
            st.info("""
            **Dashboard Features:**
            - ✅ Comprehensive data quality metrics
            - ✅ Column-level quality analysis
            - ✅ Missing values visualization
            - ✅ Outlier detection
            - ✅ Data completeness monitoring
            
            **Quality Dimensions:**
            - **Completeness**: Percentage of non-missing values
            - **Uniqueness**: Percentage of unique records
            - **Validity**: Data format and range validation
            - **Consistency**: Data consistency across records
            """)
    
    # ========================================================================
    # DATA LOADING SECTION
    # ========================================================================
    st.header("📥 Data Overview")
    
    with st.spinner("🔄 Loading data for quality analysis... Please wait."):
        df = load_data()
    
    if df is None or df.empty:
        st.error("❌ Failed to load data. Please check your data files and try again.")
        st.info("💡 **Troubleshooting:**\n"
                "- Ensure data files exist in the `data/processed/` directory\n"
                "- Check file permissions\n"
                "- Verify data file format is correct")
        st.stop()
    
    # Store in session state
    st.session_state['df'] = df
    
    # Success message
    st.success(f"✅ Data loaded successfully! **{len(df):,}** records, **{len(df.columns)}** features")
    
    st.markdown("---")
    
    # ========================================================================
    # OVERALL QUALITY METRICS SECTION
    # ========================================================================
    st.header("📊 Overall Data Quality Metrics")
    
    # Calculate quality metrics
    quality_metrics = calculate_data_quality_metrics(df)
    
    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        completeness = quality_metrics.get('completeness', 0)
        completeness_color = "normal" if completeness >= completeness_threshold else "inverse"
        st.metric(
            label="Data Completeness",
            value=f"{completeness:.2f}%",
            delta=f"{quality_metrics.get('missing_percentage', 0):.2f}% missing",
            delta_color=completeness_color,
            help="Percentage of non-missing values in the dataset"
        )
    
    with col2:
        uniqueness = quality_metrics.get('uniqueness', 0)
        uniqueness_color = "normal" if uniqueness >= uniqueness_threshold else "inverse"
        st.metric(
            label="Data Uniqueness",
            value=f"{uniqueness:.2f}%",
            delta=f"{quality_metrics.get('duplicate_percentage', 0):.2f}% duplicates",
            delta_color=uniqueness_color,
            help="Percentage of unique records in the dataset"
        )
    
    with col3:
        st.metric(
            label="Total Records",
            value=f"{quality_metrics.get('total_records', 0):,}",
            help="Total number of records in the dataset"
        )
    
    with col4:
        st.metric(
            label="Total Features",
            value=quality_metrics.get('total_features', 0),
            help="Total number of features/columns in the dataset"
        )
    
    # Quality score visualization
    st.subheader("Quality Score Overview")
    
    quality_col1, quality_col2 = st.columns(2)
    
    with quality_col1:
        # Quality score gauge chart
        overall_score = (completeness + uniqueness) / 2
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = overall_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Quality Score"},
            delta = {'reference': 90},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with quality_col2:
        # Quality breakdown
        quality_breakdown = pd.DataFrame({
            'Metric': ['Completeness', 'Uniqueness'],
            'Score': [completeness, uniqueness],
            'Status': [
                '✅ Good' if completeness >= completeness_threshold else '⚠️ Needs Attention',
                '✅ Good' if uniqueness >= uniqueness_threshold else '⚠️ Needs Attention'
            ]
        })
        
        fig = px.bar(
            quality_breakdown,
            x='Metric',
            y='Score',
            color='Status',
            title='Quality Metrics Breakdown',
            color_discrete_map={'✅ Good': '#00CC96', '⚠️ Needs Attention': '#FF6B6B'},
            text='Score'
        )
        fig.update_layout(
            template='plotly_dark',
            height=300,
            yaxis_title="Score (%)",
            showlegend=True
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # MISSING VALUES ANALYSIS
    # ========================================================================
    st.header("🔍 Missing Values Analysis")
    
    missing_tab1, missing_tab2, missing_tab3 = st.tabs([
        "📊 Overview",
        "📈 By Column",
        "🔎 Patterns"
    ])
    
    with missing_tab1:
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Missing Values Summary")
                st.metric("Columns with Missing Values", len(missing_data))
                st.metric("Total Missing Values", f"{missing_data.sum():,}")
                st.metric("Average Missing per Column", f"{missing_data.mean():.1f}")
            
            with col2:
                st.subheader("Top 10 Columns with Missing Values")
                top_missing = missing_data.head(10)
                fig, ax = plt.subplots(figsize=(10, 6))
                setup_style()
                ax.barh(top_missing.index, top_missing.values, color='#FF6B6B')
                ax.set_xlabel("Number of Missing Values")
                ax.set_title("Top 10 Columns with Missing Values")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.success("✅ **Excellent!** No missing values found in the dataset!")
    
    with missing_tab2:
        if len(missing_data) > 0:
            # Interactive missing values chart
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': (missing_data.values / len(df) * 100).round(2)
            })
            
            fig = px.bar(
                missing_df,
                x='Column',
                y='Missing Count',
                title='Missing Values by Column',
                hover_data=['Missing %'],
                color='Missing Count',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                template='plotly_dark',
                height=500,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Table view
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.info("No missing values to display.")
    
    with missing_tab3:
        st.subheader("Missing Value Patterns")
        st.info("💡 **Pattern Analysis:** Identify columns that tend to have missing values together.")
        
        # Correlation of missing values
        missing_matrix = df.isnull()
        missing_corr = missing_matrix.corr()
        
        if len(missing_corr) > 0:
            fig, ax = plt.subplots(figsize=(12, 10))
            setup_style()
            sns.heatmap(
                missing_corr,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                ax=ax
            )
            ax.set_title("Missing Values Correlation Matrix")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    
    # ========================================================================
    # COLUMN-LEVEL QUALITY REPORT
    # ========================================================================
    if show_detailed_report:
        st.header("📋 Column-Level Quality Report")
        
        column_report = get_column_quality_report(df)
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_type = st.selectbox(
                "Filter by Data Type",
                options=["All"] + list(column_report['Data Type'].unique())
            )
        with col2:
            min_completeness = st.slider(
                "Minimum Completeness (%)",
                0, 100, 0
            )
        
        # Apply filters
        filtered_report = column_report.copy()
        if filter_type != "All":
            filtered_report = filtered_report[filtered_report['Data Type'] == filter_type]
        filtered_report = filtered_report[filtered_report['Completeness %'] >= min_completeness]
        
        # Display report
        st.dataframe(
            filtered_report,
            use_container_width=True,
            height=400
        )
        
        # Download report
        csv_report = filtered_report.to_csv(index=False)
        st.download_button(
            label="📥 Download Quality Report (CSV)",
            data=csv_report,
            file_name=f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary"
        )
        
        st.markdown("---")
    
    # ========================================================================
    # DATA TYPE DISTRIBUTION
    # ========================================================================
    if show_data_types:
        st.header("📊 Data Type Distribution")
        
        dtype_counts = df.dtypes.value_counts()
        dtype_df = pd.DataFrame({
            'Data Type': dtype_counts.index.astype(str),
            'Count': dtype_counts.values
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                dtype_df,
                values='Count',
                names='Data Type',
                title='Data Type Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(dtype_df, use_container_width=True)
        
        st.markdown("---")
    
    # ========================================================================
    # OUTLIER ANALYSIS
    # ========================================================================
    if show_outlier_analysis:
        st.header("🎯 Outlier Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            outlier_col = st.selectbox(
                "Select Column for Outlier Analysis",
                options=numeric_cols,
                key="outlier_col"
            )
            
            method = st.radio(
                "Detection Method",
                options=["IQR (Interquartile Range)", "Z-Score"],
                horizontal=True
            )
            
            method_key = 'iqr' if method == "IQR (Interquartile Range)" else 'zscore'
            outliers = detect_outliers(df, outlier_col, method_key)
            
            if len(outliers) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Outliers", len(outliers))
                    st.metric("Outlier Percentage", f"{(len(outliers) / len(df) * 100):.2f}%")
                
                with col2:
                    # Box plot
                    fig = px.box(
                        df,
                        y=outlier_col,
                        title=f'Outlier Detection: {outlier_col}',
                        points='outliers'
                    )
                    fig.update_layout(template='plotly_dark', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Outlier details
                with st.expander("📋 View Outlier Records"):
                    st.dataframe(outliers[[outlier_col]], use_container_width=True)
            else:
                st.success(f"✅ No outliers detected in '{outlier_col}' using {method} method!")
        else:
            st.info("No numeric columns available for outlier analysis.")
        
        st.markdown("---")
    
    # ========================================================================
    # QUALITY RECOMMENDATIONS
    # ========================================================================
    st.header("💡 Quality Recommendations")
    
    recommendations = []
    
    if completeness < completeness_threshold:
        recommendations.append({
            'Priority': 'High',
            'Issue': 'Low Data Completeness',
            'Recommendation': f'Data completeness ({completeness:.2f}%) is below threshold ({completeness_threshold}%). Consider data imputation or investigating missing value patterns.'
        })
    
    if uniqueness < uniqueness_threshold:
        recommendations.append({
            'Priority': 'High',
            'Issue': 'Low Data Uniqueness',
            'Recommendation': f'Data uniqueness ({uniqueness:.2f}%) is below threshold ({uniqueness_threshold}%). Consider removing duplicate records or investigating data collection processes.'
        })
    
    if quality_metrics.get('columns_with_missing', 0) > 0:
        recommendations.append({
            'Priority': 'Medium',
            'Issue': 'Multiple Columns with Missing Values',
            'Recommendation': f'{quality_metrics.get("columns_with_missing", 0)} columns have missing values. Review data collection and preprocessing steps.'
        })
    
    if len(recommendations) == 0:
        st.success("✅ **Excellent Data Quality!** No critical issues detected.")
    else:
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # EXPORT SECTION
    # ========================================================================
    st.header("💾 Export Quality Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export full quality report
        full_report = get_column_quality_report(df)
        csv_full = full_report.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Quality Report (CSV)",
            data=csv_full,
            file_name=f"full_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        # Export quality summary
        summary_data = {
            'Metric': ['Completeness', 'Uniqueness', 'Total Records', 'Total Features'],
            'Value': [
                f"{completeness:.2f}%",
                f"{uniqueness:.2f}%",
                f"{quality_metrics.get('total_records', 0):,}",
                quality_metrics.get('total_features', 0)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Quality Summary (CSV)",
            data=csv_summary,
            file_name=f"quality_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.caption(
        f"💡 **Tip:** Adjust quality thresholds in the sidebar to customize quality assessment. "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    main()

