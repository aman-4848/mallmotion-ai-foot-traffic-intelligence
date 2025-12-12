# ⚡ Quick Reference Card - Streamlit Page Template

## 🚀 Quick Start

```python
# 1. Copy template
cp streamlit_app/pages/TEMPLATE_NewPage.py streamlit_app/pages/9_YourPageName.py

# 2. Update configuration
PAGE_TITLE = "🎯 Your Title"
PAGE_DESCRIPTION = "Your description"
PAGE_ICON = "🎯"

# 3. Customize and test
streamlit run streamlit_app/app.py
```

## 📋 Essential Imports

```python
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from streamlit_app.utils.data_loader import load_processed_data
from streamlit_app.utils.error_handler import handle_errors, show_error
from streamlit_app.utils.charts import setup_style
```

## 🎯 Common Patterns

### Data Loading with Caching
```python
@st.cache_data(ttl=3600)
@handle_errors
def load_data():
    df = load_processed_data()
    return df

df = load_data()
if df is None:
    st.error("Failed to load data")
    st.stop()
```

### Metrics Display
```python
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Label", value, delta="+5%")
with col2:
    st.metric("Label", value)
with col3:
    st.metric("Label", value)
```

### Tabs Organization
```python
tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])
with tab1:
    st.write("Content 1")
with tab2:
    st.write("Content 2")
```

### Sidebar Filters
```python
with st.sidebar:
    st.header("Filters")
    option = st.selectbox("Option", ["A", "B", "C"])
    date_range = st.date_input("Date Range", [])
```

### Charts (Matplotlib)
```python
setup_style()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)
ax.set_title("Title")
st.pyplot(fig)
plt.close()
```

### Charts (Plotly)
```python
import plotly.express as px
fig = px.scatter(df, x='x', y='y', color='category')
fig.update_layout(template='plotly_dark')
st.plotly_chart(fig, use_container_width=True)
```

### Data Display
```python
# DataFrame
st.dataframe(df, use_container_width=True)

# Table (static)
st.table(df.head(10))

# JSON
st.json(data_dict)
```

### Interactive Elements
```python
# Selectbox
option = st.selectbox("Choose", options)

# Multiselect
options = st.multiselect("Choose", options)

# Slider
value = st.slider("Value", 0, 100, 50)

# Checkbox
enabled = st.checkbox("Enable")

# Radio
choice = st.radio("Choice", ["A", "B", "C"])

# Button
if st.button("Click"):
    st.write("Clicked!")
```

### Error Handling
```python
try:
    result = risky_operation()
except Exception as e:
    show_error(e, context="Operation Name")
    st.stop()
```

### Download Buttons
```python
# CSV
csv = df.to_csv(index=False)
st.download_button("Download CSV", csv, "file.csv", "text/csv")

# JSON
json_str = df.to_json()
st.download_button("Download JSON", json_str, "file.json", "application/json")
```

### Expanders
```python
with st.expander("More Info"):
    st.write("Hidden content")
    st.dataframe(df)
```

### Spinners
```python
with st.spinner("Loading..."):
    data = load_data()
```

### Success/Error Messages
```python
st.success("✅ Success!")
st.error("❌ Error!")
st.warning("⚠️ Warning!")
st.info("ℹ️ Info!")
```

## 🎨 Styling

### Custom CSS
```python
st.markdown("""
<style>
    .custom-class {
        color: #38BDF8;
    }
</style>
""", unsafe_allow_html=True)
```

### Markdown
```python
st.markdown("**Bold** *Italic* [Link](url)")
st.markdown("---")  # Horizontal line
```

## 📊 Data Operations

### Filtering
```python
filtered_df = df[df['column'] > 100]
filtered_df = df.query("column > 100 & other_column == 'value'")
```

### Grouping
```python
grouped = df.groupby('category')['value'].mean()
```

### Aggregation
```python
summary = df.groupby('category').agg({
    'value1': 'mean',
    'value2': ['min', 'max', 'sum']
})
```

## 🔄 Session State

```python
# Set
st.session_state['key'] = value

# Get
value = st.session_state.get('key', default_value)

# Check
if 'key' in st.session_state:
    # Use value
    pass
```

## 🎯 Page Structure Template

```python
def main():
    # 1. Header
    st.title("Page Title")
    st.markdown("Description")
    st.markdown("---")
    
    # 2. Sidebar
    with st.sidebar:
        # Filters, settings, etc.
        pass
    
    # 3. Data Loading
    df = load_data()
    if df is None:
        st.error("Error")
        st.stop()
    
    # 4. Metrics
    col1, col2, col3 = st.columns(3)
    # Display metrics
    
    # 5. Visualizations
    # Charts, graphs, etc.
    
    # 6. Data Analysis
    # Tables, statistics, etc.
    
    # 7. Footer
    st.caption("Footer text")

if __name__ == "__main__":
    main()
```

## 🛠️ Utility Functions

### Load Data
```python
from streamlit_app.utils.data_loader import load_processed_data
df = load_processed_data()
```

### Load Models
```python
from streamlit_app.utils.model_loader import load_classification_model
model = load_classification_model('random_forest')
```

### Error Handling
```python
from streamlit_app.utils.error_handler import handle_errors, safe_execute

@handle_errors
def my_function():
    pass

result, error = safe_execute(risky_function, arg1, arg2)
```

### Charts
```python
from streamlit_app.utils.charts import setup_style, create_bar_chart
setup_style()
fig = create_bar_chart(df, 'x', 'y', 'Title')
st.pyplot(fig)
```

## 📝 Best Practices Checklist

- ✅ Use `@st.cache_data` for expensive operations
- ✅ Add error handling with `@handle_errors`
- ✅ Include type hints in function signatures
- ✅ Use logging for debugging
- ✅ Provide user feedback (spinners, success messages)
- ✅ Close matplotlib figures with `plt.close()`
- ✅ Use `st.stop()` when data loading fails
- ✅ Validate data before processing
- ✅ Sample large datasets for visualization
- ✅ Use consistent styling and layout

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Data not loading | Check file paths, use `@handle_errors` |
| Charts not showing | Add `plt.close()` after `st.pyplot()` |
| Slow loading | Use `@st.cache_data`, sample data |
| Memory errors | Sample large datasets |
| Import errors | Check `sys.path.append()` |

## 📚 Key Files

- **Template**: `streamlit_app/pages/TEMPLATE_NewPage.py`
- **Guide**: `streamlit_app/pages/TEMPLATE_GUIDE.md`
- **Utilities**: `streamlit_app/utils/`
- **Main App**: `streamlit_app/app.py`

---

**💡 Tip**: Keep this reference handy while developing new pages!

