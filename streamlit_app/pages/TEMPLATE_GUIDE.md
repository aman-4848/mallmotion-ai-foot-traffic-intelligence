# 📘 Professional Streamlit Page Template Guide

## Overview

This guide explains how to use the professional Streamlit page template (`TEMPLATE_NewPage.py`) to create new pages for the Mall Movement Tracking dashboard.

## 🚀 Quick Start

### Step 1: Copy the Template

```bash
# Copy the template file
cp streamlit_app/pages/TEMPLATE_NewPage.py streamlit_app/pages/9_YourPageName.py
```

### Step 2: Update Page Configuration

Edit the following constants at the top of your new file:

```python
PAGE_TITLE = "🎯 Your Page Title"
PAGE_DESCRIPTION = """
Brief description of what this page does and its purpose.
"""
PAGE_ICON = "🎯"  # Choose an appropriate emoji
```

### Step 3: Customize Sections

Remove or modify sections based on your needs:
- Data Loading
- Key Metrics
- Visualizations
- Data Analysis
- Interactive Features
- Export & Sharing

### Step 4: Test Your Page

```bash
streamlit run streamlit_app/app.py
```

## 📋 Template Structure

### 1. **Imports Section**
- All necessary imports with type hints
- Project utilities (data_loader, error_handler, charts, etc.)
- Logging setup

### 2. **Configuration Section**
- Page title, description, and icon
- Constants for directories
- Configuration variables

### 3. **Helper Functions**
- `load_data()`: Cached data loading with error handling
- `load_results()`: Load JSON results files
- `calculate_metrics()`: Calculate key metrics
- `create_professional_chart()`: Create styled charts

### 4. **Main Function Sections**

#### **Sidebar Configuration**
- Page settings
- Filters
- Display options
- Information expanders

#### **Data Loading**
- Cached data loading
- Error handling
- Success messages
- Quick data info

#### **Key Metrics**
- Display important metrics
- Metric cards with help text
- Delta indicators

#### **Visualizations**
- Multiple tabs for different chart types
- Interactive Plotly charts
- Static matplotlib charts
- Chart customization options

#### **Data Analysis**
- Data preview with filtering
- Statistical summary
- Data quality report
- Missing values visualization

#### **Interactive Features**
- Model integration examples
- Custom analysis tools
- Query builders

#### **Export & Sharing**
- CSV export
- JSON export
- Excel export (optional)

#### **Additional Information**
- Documentation
- Technical details
- Known limitations
- Support information

## 🎨 Best Practices

### 1. **Error Handling**
Always use the error handling utilities:

```python
from streamlit_app.utils.error_handler import handle_errors, show_error

@handle_errors
def your_function():
    # Your code here
    pass
```

### 2. **Caching**
Use `@st.cache_data` for expensive operations:

```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def expensive_operation():
    # Expensive computation
    pass
```

### 3. **Type Hints**
Always include type hints for better code clarity:

```python
def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    # Function implementation
    pass
```

### 4. **Logging**
Use logging for debugging and monitoring:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Operation started")
logger.error("Error occurred", exc_info=True)
```

### 5. **User Feedback**
Provide clear feedback to users:

```python
with st.spinner("Loading data..."):
    data = load_data()

if data is not None:
    st.success("✅ Data loaded successfully!")
else:
    st.error("❌ Failed to load data")
```

## 🔧 Customization Examples

### Example 1: Simple Metrics Page

```python
def main():
    st.title("📊 Simple Metrics")
    
    df = load_data()
    if df is None:
        st.error("Failed to load data")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Total Features", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

### Example 2: Model Comparison Page

```python
def main():
    st.title("🤖 Model Comparison")
    
    # Load model results
    results = load_results(RESULTS_DIR / "classification" / "metrics.json")
    
    if results:
        # Display model metrics
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        
        fig = create_professional_chart(
            'bar',
            pd.DataFrame({'Model': models, 'Accuracy': accuracies}),
            'Model',
            'Accuracy',
            'Model Accuracy Comparison'
        )
        st.pyplot(fig)
```

### Example 3: Interactive Dashboard

```python
def main():
    st.title("📈 Interactive Dashboard")
    
    df = load_data()
    
    # Sidebar filters
    with st.sidebar:
        selected_zones = st.multiselect("Select Zones", df['zone'].unique())
        date_range = st.date_input("Date Range", [])
    
    # Apply filters
    filtered_df = df[df['zone'].isin(selected_zones)]
    
    # Interactive chart
    fig = px.scatter(
        filtered_df,
        x='feature1',
        y='feature2',
        color='zone',
        size='value'
    )
    st.plotly_chart(fig, use_container_width=True)
```

## 📦 Available Utilities

### Data Loading
- `load_processed_data()`: Load processed dataset
- `get_data_info()`: Get data information
- `validate_dataframe()`: Validate DataFrame structure

### Error Handling
- `@handle_errors`: Decorator for error handling
- `safe_execute()`: Safe function execution
- `show_error()`: Display user-friendly errors

### Charts
- `setup_style()`: Setup dark theme styling
- `create_bar_chart()`: Create bar charts
- `create_line_plot()`: Create line charts
- `create_heatmap()`: Create heatmap visualizations

### Model Loading
- `load_classification_model()`: Load classification models
- `load_forecasting_model()`: Load forecasting models

## 🎯 Common Patterns

### Pattern 1: Conditional Rendering

```python
if condition:
    st.success("✅ Success message")
    # Show content
else:
    st.warning("⚠️ Warning message")
    st.stop()  # Stop execution
```

### Pattern 2: Tabs Organization

```python
tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])

with tab1:
    # Content for tab 1
    pass

with tab2:
    # Content for tab 2
    pass
```

### Pattern 3: Columns Layout

```python
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Metric 1", value1)

with col2:
    st.metric("Metric 2", value2)

with col3:
    st.metric("Metric 3", value3)
```

### Pattern 4: Expanders for Details

```python
with st.expander("📚 More Information"):
    st.markdown("""
    Detailed information here.
    - Point 1
    - Point 2
    - Point 3
    """)
```

## 🐛 Troubleshooting

### Issue: Data not loading
**Solution:** Check file paths and ensure data files exist in the correct directories.

### Issue: Charts not displaying
**Solution:** Ensure matplotlib figures are properly closed with `plt.close()`.

### Issue: Slow page loading
**Solution:** Use caching (`@st.cache_data`) for expensive operations.

### Issue: Memory errors
**Solution:** Sample large datasets before visualization.

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Plotly Documentation](https://plotly.com/python/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## ✅ Checklist for New Pages

- [ ] Copy template file
- [ ] Update page configuration (title, description, icon)
- [ ] Customize sections based on needs
- [ ] Add appropriate error handling
- [ ] Implement caching for expensive operations
- [ ] Add type hints to functions
- [ ] Test with various data scenarios
- [ ] Add logging statements
- [ ] Test error scenarios
- [ ] Update documentation
- [ ] Test page in full dashboard context

## 🎉 Tips for Success

1. **Start Simple**: Begin with basic functionality, then add features incrementally
2. **Test Early**: Test your page frequently during development
3. **Use Utilities**: Leverage existing utilities instead of reinventing
4. **Follow Patterns**: Use consistent patterns across pages
5. **Document Code**: Add comments and docstrings for clarity
6. **Handle Errors**: Always include proper error handling
7. **User Experience**: Focus on clear, intuitive user interface
8. **Performance**: Optimize for speed and responsiveness

---

**Happy Coding! 🚀**

For questions or issues, refer to the main project documentation or contact the development team.

