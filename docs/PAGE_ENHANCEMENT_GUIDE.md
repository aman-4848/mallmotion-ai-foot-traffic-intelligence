# Page Enhancement Guide - Quick Reference

## 🚀 Quick Start: Adding Functionality to Existing Pages

### 1. Add Filtering to Data Explorer Page

**File**: `streamlit_app/pages/2_Data_Explorer.py`

**Add after line 30**:
```python
# Add sidebar filters
st.sidebar.header("🔍 Filters")

# Date filter (if datetime column exists)
datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
if datetime_cols:
    date_col = st.sidebar.selectbox("Select Date Column", datetime_cols)
    date_range = st.sidebar.date_input("Date Range", value=[])
    if date_range:
        df = df[(df[date_col].dt.date >= date_range[0]) & 
                (df[date_col].dt.date <= date_range[1])]

# Zone filter
zone_cols = [col for col in df.columns if 'zone' in col.lower() or 'space' in col.lower()]
if zone_cols:
    selected_zones = st.sidebar.multiselect("Select Zones", df[zone_cols[0]].unique())
    if selected_zones:
        df = df[df[zone_cols[0]].isin(selected_zones)]
```

---

### 2. Add Export to Classification Results Page

**File**: `streamlit_app/pages/4_Classification_Results.py`

**Add after the results table (around line 163)**:
```python
# Export results
if st.button("📥 Export Results"):
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="classification_results.csv",
        mime="text/csv"
    )
```

---

### 3. Add Interactive Charts to Overview Page

**File**: `streamlit_app/pages/1_Overview.py`

**Add after metrics section**:
```python
# Add interactive chart
import plotly.express as px

st.header("📈 Performance Trends")

# Load historical data if available
try:
    # Create sample trend data
    trend_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
        'Accuracy': np.random.uniform(0.7, 0.95, 30)
    })
    
    fig = px.line(trend_data, x='Date', y='Accuracy', 
                  title='Model Accuracy Over Time',
                  labels={'Accuracy': 'Accuracy Score'})
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not load trend data: {e}")
```

---

### 4. Add Prediction History to Predict Next Zone Page

**File**: `streamlit_app/pages/7_Predict_Next_Zone.py`

**Add at the top (after imports)**:
```python
# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
```

**Add after prediction (around line 95)**:
```python
# Save to history
st.session_state.prediction_history.append({
    'timestamp': pd.Timestamp.now(),
    'model': model_choice,
    'prediction': prediction,
    'confidence': probabilities.max() if probabilities is not None else None
})

# Display history
if st.checkbox("Show Prediction History"):
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True)
    
    # Clear history button
    if st.button("Clear History"):
        st.session_state.prediction_history = []
        st.rerun()
```

---

### 5. Add Time Range Selector to Heatmaps Page

**File**: `streamlit_app/pages/3_Heatmaps.py`

**Add after data loading (around line 37)**:
```python
# Time range selector
st.sidebar.header("⏰ Time Filters")

temporal_cols = [col for col in df.columns if any(x in col.lower() for x in ['hour', 'day', 'week', 'month', 'date', 'time'])]
if temporal_cols:
    time_col = st.sidebar.selectbox("Select Time Column", temporal_cols)
    
    if df[time_col].dtype in ['datetime64[ns]', 'object']:
        try:
            if df[time_col].dtype == 'object':
                df[time_col] = pd.to_datetime(df[time_col])
            
            min_date = df[time_col].min().date()
            max_date = df[time_col].max().date()
            
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                df = df[(df[time_col].dt.date >= date_range[0]) & 
                        (df[time_col].dt.date <= date_range[1])]
        except Exception as e:
            st.sidebar.warning(f"Could not filter by date: {e}")
```

---

### 6. Add Cluster Characteristics to Clustering Page

**File**: `streamlit_app/pages/5_Clustering_Insights.py`

**Add after cluster visualization (around line 150)**:
```python
# Cluster characteristics
st.header("🔍 Cluster Characteristics")

if 'labels' in locals():
    # Add cluster labels to dataframe
    df['cluster'] = labels
    
    # Analyze each cluster
    cluster_analysis = []
    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id == -1:
            continue  # Skip noise
        
        cluster_data = df[df['cluster'] == cluster_id]
        
        # Calculate characteristics
        characteristics = {
            'Cluster': f'C{cluster_id}',
            'Size': len(cluster_data),
            'Avg Visits': cluster_data.groupby('USERID').size().mean() if 'USERID' in df.columns else 0,
            'Top Zone': cluster_data['SPACEID'].mode()[0] if 'SPACEID' in df.columns else 'N/A',
        }
        
        # Add numeric feature averages
        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols[:5]:  # Top 5 numeric features
            if col not in ['USERID', 'SPACEID', 'cluster']:
                characteristics[f'Avg {col}'] = cluster_data[col].mean()
        
        cluster_analysis.append(characteristics)
    
    if cluster_analysis:
        cluster_df = pd.DataFrame(cluster_analysis)
        st.dataframe(cluster_df, use_container_width=True)
```

---

### 7. Add Forecast Visualization to Forecasting Page

**File**: `streamlit_app/pages/6_Forecasting_Traffic.py`

**Add after model comparison (around line 69)**:
```python
# Forecast visualization
st.header("📊 Forecast Visualization")

try:
    # Load forecasting model
    prophet_path = models_dir / "forecasting" / "prophet_model.pkl"
    if prophet_path.exists():
        model = joblib.load(prophet_path)
        
        # Generate future forecast
        future_periods = st.slider("Forecast Periods", 7, 90, 30)
        
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                # Create future dataframe (adjust based on your Prophet model structure)
                # This is a template - adjust based on your actual model
                future = model.make_future_dataframe(periods=future_periods)
                forecast = model.predict(future)
                
                # Plot forecast
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=forecast['ds'][:-future_periods],
                    y=forecast['yhat'][:-future_periods],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast['ds'][-future_periods:],
                    y=forecast['yhat'][-future_periods:],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast['ds'][-future_periods:],
                    y=forecast['yhat_upper'][-future_periods:],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast['ds'][-future_periods:],
                    y=forecast['yhat_lower'][-future_periods:],
                    mode='lines',
                    name='Confidence Interval',
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(width=0)
                ))
                
                fig.update_layout(
                    title='Traffic Forecast',
                    xaxis_title='Date',
                    yaxis_title='Traffic',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not generate forecast visualization: {e}")
```

---

## 📝 Common Patterns

### Pattern 1: Add Caching
```python
@st.cache_data
def expensive_operation():
    # Your expensive operation
    return result
```

### Pattern 2: Add Loading State
```python
with st.spinner("Processing..."):
    result = process_data()
st.success("Done!")
```

### Pattern 3: Add Error Handling
```python
try:
    result = risky_operation()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()
```

### Pattern 4: Add Refresh Button
```python
if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()
```

### Pattern 5: Add Download Button
```python
csv = df.to_csv(index=False)
st.download_button(
    label="📥 Download",
    data=csv,
    file_name="data.csv",
    mime="text/csv"
)
```

---

## 🎯 Quick Enhancement Checklist

When enhancing a page:

- [ ] Add error handling
- [ ] Add loading states
- [ ] Add caching for performance
- [ ] Add export functionality
- [ ] Add filters/interactivity
- [ ] Add visualizations
- [ ] Test all functionality
- [ ] Verify styling consistency
- [ ] Update documentation

---

**Need Help?** Check `docs/STREAMLIT_PAGES_STATUS.md` for detailed page status and templates.

