# Production Features Implementation Guide

## Quick Start

This guide helps you implement production-standard features step by step.

## Phase 1: Foundation (Week 1-2)

### Step 1: Setup Configuration Management

```python
# Use the config.py module
from streamlit_app.config import config

# Access configuration
data_file = config.DATA_FILE
cache_ttl = config.DATA_CACHE_TTL
```

### Step 2: Implement Caching

```python
# In your pages, use caching decorators
from streamlit_app.utils.cache_utils import cache_data, cache_model

@cache_data(ttl=3600)  # Cache for 1 hour
def load_processed_data():
    # Your data loading code
    pass

# Cache models
model = cache_model("models/classification/zone_xgb.pkl")
```

### Step 3: Add Error Handling

```python
# Wrap functions with error handling
from streamlit_app.utils.error_handler import handle_errors, show_error

@handle_errors
def your_function():
    # Your code here
    pass

# Or use try-except with show_error
try:
    result = risky_operation()
except Exception as e:
    show_error(e, context="Loading data")
```

### Step 4: Add Input Validation

```python
# Validate inputs before processing
from streamlit_app.utils.validation import (
    validate_dataframe,
    validate_file_upload,
    validate_model_input
)

# Validate DataFrame
is_valid, error = validate_dataframe(df, required_columns=['SPACEID', 'USERID'])
if not is_valid:
    st.error(error)
    st.stop()

# Validate file upload
is_valid, error = validate_file_upload(
    uploaded_file, 
    allowed_extensions=['csv', 'xlsx'],
    max_size_mb=50
)
```

## Phase 2: User Experience (Week 3-4)

### Step 1: Add Loading States

```python
# Show progress for long operations
with st.spinner("Loading data..."):
    df = load_data()

# Or use progress bar
progress_bar = st.progress(0)
for i in range(100):
    # Your operation
    progress_bar.progress(i + 1)
```

### Step 2: Add Search Functionality

```python
# Add search box
search_term = st.text_input("🔍 Search", "")

# Filter data
if search_term:
    filtered_df = df[df['column'].str.contains(search_term, case=False)]
else:
    filtered_df = df
```

### Step 3: Add Advanced Filtering

```python
# Multi-criteria filters
col1, col2, col3 = st.columns(3)

with col1:
    zone_filter = st.multiselect("Filter by Zone", df['ZONE'].unique())

with col2:
    date_range = st.date_input("Date Range", [])

with col3:
    user_filter = st.selectbox("User", df['USERID'].unique())

# Apply filters
filtered_df = df
if zone_filter:
    filtered_df = filtered_df[filtered_df['ZONE'].isin(zone_filter)]
```

## Phase 3: Advanced Features (Week 5-6)

### Step 1: Add Export Functionality

```python
# Export data
if st.button("📥 Export Data"):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="data.csv",
        mime="text/csv"
    )

# Export visualization
if st.button("📊 Export Chart"):
    fig.savefig("chart.png")
    with open("chart.png", "rb") as file:
        st.download_button(
            label="Download PNG",
            data=file,
            file_name="chart.png",
            mime="image/png"
        )
```

### Step 2: Add Interactive Charts

```python
# Install plotly: pip install plotly
import plotly.express as px

# Create interactive chart
fig = px.scatter(df, x='x_col', y='y_col', color='category')
st.plotly_chart(fig, use_container_width=True)
```

### Step 3: Add Model Versioning

```python
# Track model versions
import json
from datetime import datetime

model_info = {
    "version": "1.0.0",
    "timestamp": datetime.now().isoformat(),
    "accuracy": 0.9965,
    "features": feature_list
}

# Save model metadata
with open("models/classification/model_info.json", 'w') as f:
    json.dump(model_info, f, indent=2)
```

## Phase 4: Security & Authentication (Week 7-8)

### Step 1: Add Basic Authentication

```python
# Simple password protection
import streamlit_authenticator as stauth

# In your app.py
authenticator = stauth.Authenticate(
    credentials,
    'cookie_name',
    'signature_key',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status == False:
    st.error('Username/password is incorrect')
    st.stop()
elif authentication_status == None:
    st.warning('Please enter your username and password')
    st.stop()
```

### Step 2: Add Session Management

```python
# Track user sessions
import time

if 'last_activity' not in st.session_state:
    st.session_state.last_activity = time.time()

# Check session timeout
if time.time() - st.session_state.last_activity > config.SESSION_TIMEOUT:
    st.warning("Session expired. Please refresh.")
    st.stop()
```

## Best Practices

### 1. Always Use Caching for Expensive Operations
```python
@st.cache_data(ttl=3600)
def expensive_operation():
    # Your code
    pass
```

### 2. Validate All User Inputs
```python
# Never trust user input
user_input = st.text_input("Enter value")
if not validate_input(user_input, validation_func, "Invalid input"):
    st.stop()
```

### 3. Handle Errors Gracefully
```python
# Always wrap risky operations
try:
    result = risky_operation()
except Exception as e:
    show_error(e, context="Operation name")
    st.stop()
```

### 4. Log Important Operations
```python
from streamlit_app.utils.error_handler import log_operation

@log_operation("Data Loading")
def load_data():
    # Your code
    pass
```

### 5. Use Configuration for Settings
```python
# Don't hardcode values
from streamlit_app.config import config

max_rows = config.MAX_ROWS_DISPLAY
cache_ttl = config.DATA_CACHE_TTL
```

## Testing Checklist

- [ ] All pages load without errors
- [ ] Caching works correctly
- [ ] Error messages are user-friendly
- [ ] Input validation works
- [ ] Export functionality works
- [ ] Performance is acceptable (< 2s load time)
- [ ] Mobile responsiveness
- [ ] Accessibility (keyboard navigation)

## Next Steps

1. Review the Production Improvement Plan
2. Prioritize features based on your needs
3. Start with Phase 1 (Foundation)
4. Test thoroughly before moving to next phase
5. Gather user feedback
6. Iterate and improve

