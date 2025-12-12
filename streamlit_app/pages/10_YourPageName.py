"""
============================================================================
PROFESSIONAL STREAMLIT PAGE TEMPLATE
============================================================================
Template for Creating New Streamlit Pages

INSTRUCTIONS:
1. Copy this file and rename it (e.g., 9_YourPageName.py)
2. Update the PAGE_CONFIG section with your page details
3. Customize the sections according to your needs
4. Remove unused sections or add new ones as needed
5. Test thoroughly before deployment

BEST PRACTICES:
- Use type hints for better code clarity
- Implement proper error handling
- Use caching for expensive operations
- Follow the existing app's styling and structure
- Add comprehensive documentation
- Test with various data scenarios
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
from streamlit_app.utils.model_loader import load_classification_model, load_forecasting_model
from streamlit_app.utils.validation import validate_dataframe
from features.feature_engineering import FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
PAGE_TITLE = "🎯 Your Page Title"
PAGE_DESCRIPTION = """
Brief description of what this page does and its purpose.
This will be displayed at the top of the page.
"""
PAGE_ICON = "🎯"  # Choose an appropriate emoji icon

# ============================================================================
# CONSTANTS
# ============================================================================
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

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
        logger.info("Loading processed data...")
        df = load_processed_data()
        
        if df is None or df.empty:
            raise ValueError("Data is empty or None")
        
        # Apply feature engineering if needed
        fe = FeatureEngineer()
        df = fe.engineer_features(df)
        
        # Validate data
        is_valid, error_message = validate_dataframe(df)
        if not is_valid and error_message:
            logger.warning(f"Data validation warnings: {error_message}")
        
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


@st.cache_data(ttl=3600)
def load_results(file_path: Path) -> Optional[Dict]:
    """
    Load JSON results file with caching.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with results or None if loading fails
    """
    try:
        if not file_path.exists():
            logger.warning(f"Results file not found: {file_path}")
            return None
        
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading results from {file_path}: {e}")
        return None


@handle_errors
def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate key metrics from the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with calculated metrics
    """
    if df is None or df.empty:
        return {}
    
    metrics = {
        'total_records': len(df),
        'total_features': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Add numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        metrics['numeric_features'] = len(numeric_cols)
        metrics['mean_values'] = df[numeric_cols].mean().to_dict()
    
    return metrics


def create_professional_chart(
    chart_type: str,
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Create a professional chart with consistent styling.
    
    Args:
        chart_type: Type of chart ('bar', 'line', 'scatter', 'hist')
        data: DataFrame with data
        x: X-axis column name
        y: Y-axis column name
        title: Chart title
        color: Color for the chart
        **kwargs: Additional arguments
        
    Returns:
        Matplotlib figure
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if chart_type == 'bar':
        ax.bar(data[x], data[y], color=color or '#38BDF8', alpha=0.8)
    elif chart_type == 'line':
        ax.plot(data[x], data[y], color=color or '#38BDF8', linewidth=2, marker='o')
    elif chart_type == 'scatter':
        ax.scatter(data[x], data[y], color=color or '#38BDF8', alpha=0.6)
    elif chart_type == 'hist':
        ax.hist(data[x], bins=kwargs.get('bins', 30), color=color or '#38BDF8', alpha=0.7)
    
    ax.set_title(title, fontsize=16, fontweight='bold', color='white')
    ax.set_xlabel(x, fontsize=12, color='white')
    ax.set_ylabel(y, fontsize=12, color='white')
    ax.grid(True, alpha=0.3, color='gray')
    
    plt.tight_layout()
    return fig


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
        st.header("⚙️ Page Settings")
        
        # Data refresh option
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Filters section
        st.subheader("🔍 Filters")
        enable_filters = st.checkbox("Enable Filters", value=False)
        
        if enable_filters:
            # Date range filter
            date_col = st.selectbox(
                "Select Date Column",
                options=["None"] + list(st.session_state.get('df', pd.DataFrame()).columns),
                help="Select a date column for filtering"
            )
            
            if date_col and date_col != "None":
                # Get date range from data
                df_temp = st.session_state.get('df', pd.DataFrame())
                if not df_temp.empty and date_col in df_temp.columns:
                    min_date = pd.to_datetime(df_temp[date_col]).min().date()
                    max_date = pd.to_datetime(df_temp[date_col]).max().date()
                    
                    date_range = st.date_input(
                        "Select Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
        
        st.markdown("---")
        
        # Display options
        st.subheader("📊 Display Options")
        show_raw_data = st.checkbox("Show Raw Data", value=False)
        show_statistics = st.checkbox("Show Statistics", value=True)
        chart_theme = st.selectbox(
            "Chart Theme",
            options=["Dark", "Light", "Auto"],
            index=0
        )
        
        st.markdown("---")
        
        # Information section
        with st.expander("ℹ️ About This Page"):
            st.info("""
            **Page Purpose:**
            - Feature 1: Description
            - Feature 2: Description
            - Feature 3: Description
            
            **Data Source:** Processed dataset
            **Last Updated:** Auto-refreshed
            """)
    
    # ========================================================================
    # DATA LOADING SECTION
    # ========================================================================
    st.header("📥 Data Loading")
    
    with st.spinner("🔄 Loading data... Please wait."):
        df = load_data()
    
    if df is None or df.empty:
        st.error("❌ Failed to load data. Please check your data files and try again.")
        st.info("💡 **Troubleshooting:**\n"
                "- Ensure data files exist in the `data/processed/` directory\n"
                "- Check file permissions\n"
                "- Verify data file format is correct")
        st.stop()
    
    # Store in session state for filters
    st.session_state['df'] = df
    
    # Success message
    st.success(f"✅ Data loaded successfully! **{len(df):,}** records, **{len(df.columns)}** features")
    
    # Quick data info
    with st.expander("📋 Quick Data Info", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    st.markdown("---")
    
    # ========================================================================
    # KEY METRICS SECTION
    # ========================================================================
    st.header("📊 Key Metrics")
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Records",
            value=f"{metrics.get('total_records', 0):,}",
            delta=None,
            help="Total number of records in the dataset"
        )
    
    with col2:
        st.metric(
            label="Total Features",
            value=metrics.get('total_features', 0),
            delta=None,
            help="Number of features/columns in the dataset"
        )
    
    with col3:
        missing_pct = (metrics.get('missing_values', 0) / 
                      (metrics.get('total_records', 1) * metrics.get('total_features', 1)) * 100)
        st.metric(
            label="Missing Values",
            value=f"{metrics.get('missing_values', 0):,}",
            delta=f"{missing_pct:.2f}%",
            delta_color="inverse",
            help="Total missing values in the dataset"
        )
    
    with col4:
        st.metric(
            label="Memory Usage",
            value=f"{metrics.get('memory_usage_mb', 0):.2f} MB",
            delta=None,
            help="Memory usage of the dataset"
        )
    
    st.markdown("---")
    
    # ========================================================================
    # VISUALIZATIONS SECTION
    # ========================================================================
    st.header("📈 Visualizations")
    
    # Tabs for different visualization types
    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "📊 Overview Charts",
        "📈 Time Series",
        "🎯 Interactive Charts"
    ])
    
    with viz_tab1:
        st.subheader("Overview Statistics")
        
        # Select columns for visualization
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox(
                "Select X-axis Column",
                options=df.columns.tolist(),
                key="x_col_overview"
            )
        
        with col2:
            y_col = st.selectbox(
                "Select Y-axis Column",
                options=df.select_dtypes(include=[np.number]).columns.tolist(),
                key="y_col_overview"
            )
        
        # Chart type selection
        chart_type = st.radio(
            "Select Chart Type",
            options=["Bar Chart", "Line Chart", "Scatter Plot", "Histogram"],
            horizontal=True
        )
        
        # Generate chart
        if st.button("📊 Generate Chart", type="primary"):
            try:
                chart_type_lower = chart_type.lower().replace(" ", "_")
                
                if chart_type_lower == "histogram":
                    fig = create_professional_chart(
                        'hist',
                        df,
                        x_col,
                        y_col,
                        f"{chart_type}: {x_col}",
                        color='#38BDF8'
                    )
                else:
                    # Aggregate data if needed
                    if chart_type_lower == "bar_chart":
                        chart_data = df.groupby(x_col)[y_col].mean().reset_index()
                    else:
                        chart_data = df[[x_col, y_col]].copy()
                    
                    fig = create_professional_chart(
                        chart_type_lower.replace("_", ""),
                        chart_data,
                        x_col,
                        y_col,
                        f"{chart_type}: {x_col} vs {y_col}",
                        color='#38BDF8'
                    )
                
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                show_error(e, context="Chart Generation")
    
    with viz_tab2:
        st.subheader("Time Series Analysis")
        
        # Time series visualization using Plotly
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            ts_col = st.selectbox(
                "Select Column for Time Series",
                options=numeric_cols,
                key="ts_col"
            )
            
            # Sample data for demonstration
            sample_size = min(1000, len(df))
            df_sample = df.sample(n=sample_size).sort_index()
            
            fig = px.line(
                df_sample,
                x=df_sample.index,
                y=ts_col,
                title=f"Time Series: {ts_col}",
                labels={'index': 'Index', ts_col: ts_col}
            )
            fig.update_layout(
                template='plotly_dark',
                height=500,
                xaxis_title="Index",
                yaxis_title=ts_col
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for time series analysis.")
    
    with viz_tab3:
        st.subheader("Interactive Visualizations")
        
        # Interactive scatter plot
        col1, col2, col3 = st.columns(3)
        
        with col1:
            scatter_x = st.selectbox("X-axis", options=df.select_dtypes(include=[np.number]).columns.tolist())
        with col2:
            scatter_y = st.selectbox("Y-axis", options=df.select_dtypes(include=[np.number]).columns.tolist())
        with col3:
            color_by = st.selectbox("Color By", options=["None"] + df.columns.tolist())
        
        if scatter_x and scatter_y:
            fig = px.scatter(
                df.sample(n=min(1000, len(df))),
                x=scatter_x,
                y=scatter_y,
                color=color_by if color_by != "None" else None,
                title=f"Interactive Scatter: {scatter_x} vs {scatter_y}",
                hover_data=df.columns.tolist()[:5]  # Show first 5 columns in hover
            )
            fig.update_layout(template='plotly_dark', height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # DATA ANALYSIS SECTION
    # ========================================================================
    st.header("🔍 Data Analysis")
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
        "📋 Data Preview",
        "📊 Statistical Summary",
        "🔎 Data Quality"
    ])
    
    with analysis_tab1:
        st.subheader("Data Preview")
        
        # Number of rows to display
        n_rows = st.slider("Number of Rows to Display", 10, 1000, 100, 10)
        
        # Column selection
        selected_cols = st.multiselect(
            "Select Columns to Display",
            options=df.columns.tolist(),
            default=df.columns.tolist()[:10]  # First 10 columns by default
        )
        
        if selected_cols:
            st.dataframe(
                df[selected_cols].head(n_rows),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = df[selected_cols].to_csv(index=False)
            st.download_button(
                label="📥 Download Selected Data (CSV)",
                data=csv,
                file_name=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.warning("Please select at least one column to display.")
    
    with analysis_tab2:
        st.subheader("Statistical Summary")
        
        if show_statistics:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                selected_numeric = st.multiselect(
                    "Select Numeric Columns",
                    options=numeric_cols,
                    default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
                )
                
                if selected_numeric:
                    st.dataframe(
                        df[selected_numeric].describe(),
                        use_container_width=True
                    )
                else:
                    st.info("Please select numeric columns to view statistics.")
            else:
                st.info("No numeric columns available for statistical analysis.")
    
    with analysis_tab3:
        st.subheader("Data Quality Report")
        
        # Data quality metrics
        quality_metrics = {
            "Completeness": f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%",
            "Uniqueness": f"{(1 - df.duplicated().sum() / len(df)) * 100:.2f}%",
            "Validity": "100.00%",  # Placeholder - implement actual validation
        }
        
        for metric, value in quality_metrics.items():
            st.metric(metric, value)
        
        # Missing values visualization
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            st.subheader("Missing Values by Column")
            fig, ax = plt.subplots(figsize=(10, 6))
            setup_style()
            ax.barh(missing_data.index[:20], missing_data.values[:20], color='#FF6B6B')
            ax.set_xlabel("Number of Missing Values")
            ax.set_title("Top 20 Columns with Missing Values")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.success("✅ No missing values found in the dataset!")
    
    st.markdown("---")
    
    # ========================================================================
    # INTERACTIVE FEATURES SECTION
    # ========================================================================
    st.header("🎮 Interactive Features")
    
    feature_tab1, feature_tab2 = st.tabs([
        "🔧 Model Integration",
        "⚙️ Custom Analysis"
    ])
    
    with feature_tab1:
        st.subheader("Model Predictions")
        
        # Model selection
        model_type = st.radio(
            "Select Model Type",
            options=["Classification", "Forecasting"],
            horizontal=True
        )
        
        if model_type == "Classification":
            model_choice = st.selectbox(
                "Select Classification Model",
                options=["Random Forest", "Decision Tree", "XGBoost", "Logistic Regression"]
            )
            
            if st.button("🔮 Make Prediction", type="primary"):
                st.info("💡 Model prediction functionality - implement based on your needs")
                # Example: Load model and make prediction
                # model = load_classification_model(model_choice.lower().replace(" ", "_"))
                # prediction = model.predict(input_data)
        
        else:  # Forecasting
            model_choice = st.selectbox(
                "Select Forecasting Model",
                options=["Random Forest Regressor", "ARIMA"]
            )
            
            if st.button("📈 Forecast", type="primary"):
                st.info("💡 Forecasting functionality - implement based on your needs")
    
    with feature_tab2:
        st.subheader("Custom Analysis")
        
        # Custom query/filter
        st.write("**Custom Data Filter**")
        filter_query = st.text_input(
            "Enter Filter Query (Pandas syntax)",
            placeholder="Example: df['column'] > 100",
            help="Use pandas DataFrame filtering syntax"
        )
        
        if filter_query:
            try:
                # Safe evaluation (in production, use more secure methods)
                filtered_df = df.query(filter_query.replace('df', '').strip())
                st.success(f"✅ Filter applied! **{len(filtered_df)}** records match the criteria.")
                st.dataframe(filtered_df.head(100), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Invalid filter query: {e}")
                st.info("💡 Example valid queries:\n"
                       "- `column_name > 100`\n"
                       "- `column_name == 'value'`\n"
                       "- `(column1 > 50) & (column2 < 200)`")
    
    st.markdown("---")
    
    # ========================================================================
    # EXPORT & SHARING SECTION
    # ========================================================================
    st.header("💾 Export & Sharing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # JSON Export
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        # Excel Export (requires openpyxl)
        try:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 Download Excel",
                data=excel_data,
                file_name=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.info("💡 Install openpyxl for Excel export: `pip install openpyxl`")
    
    st.markdown("---")
    
    # ========================================================================
    # ADDITIONAL INFORMATION SECTION
    # ========================================================================
    st.header("💡 Additional Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        with st.expander("📚 Documentation", expanded=False):
            st.markdown("""
            **Page Features:**
            - ✅ Data loading with caching
            - ✅ Interactive visualizations
            - ✅ Statistical analysis
            - ✅ Data export functionality
            - ✅ Model integration support
            
            **Usage Tips:**
            - Use sidebar filters to customize your view
            - Export data in multiple formats
            - Explore different visualization types
            - Check data quality metrics
            """)
        
        with st.expander("🔧 Technical Details", expanded=False):
            st.code("""
            # Key Technologies:
            - Streamlit for UI
            - Pandas for data processing
            - Matplotlib/Seaborn for static charts
            - Plotly for interactive charts
            - Custom utilities for error handling
            """, language="python")
    
    with info_col2:
        with st.expander("⚠️ Known Limitations", expanded=False):
            st.warning("""
            **Current Limitations:**
            - Large datasets may take time to load
            - Some visualizations limited to 1000 samples
            - Model predictions require pre-trained models
            
            **Future Improvements:**
            - Real-time data updates
            - Advanced filtering options
            - More visualization types
            """)
        
        with st.expander("📞 Support", expanded=False):
            st.info("""
            **Need Help?**
            - Check the documentation
            - Review error messages
            - Contact the development team
            
            **Report Issues:**
            - Include error messages
            - Describe steps to reproduce
            - Attach sample data if possible
            """)
    
    st.markdown("---")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.caption(
        f"💡 **Tip:** Use the sidebar to customize filters and display options. "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    main()
