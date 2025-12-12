"""
Mall Movement Tracking - Streamlit Dashboard
Production-ready dashboard for ML model visualization and predictions
"""
import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Mall Movement Tracking Dashboard",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, clean design - Dark theme
st.markdown("""
    <style>
    /* Color Variables */
    :root {
        --main-bg: #0F172A;
        --sidebar-bg: #1E293B;
        --text-color: #FFFFFF;
        --accent-color: #38BDF8;
        --card-bg: #1A2238;
    }
    
    /* Main app styling - Dark Slate background */
    .main {
        background-color: var(--main-bg) !important;
        color: var(--text-color) !important;
    }
    
    /* Main content area */
    .block-container {
        background-color: var(--main-bg) !important;
        color: var(--text-color) !important;
    }
    
    /* Sidebar styling - Dark Blue Grey background */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 2px solid var(--accent-color);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: var(--sidebar-bg) !important;
    }
    
    /* Sidebar text color - White */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] strong {
        color: var(--text-color) !important;
    }
    
    /* Sidebar navigation links */
    [data-testid="stSidebar"] [data-baseweb="list"] {
        background-color: var(--sidebar-bg) !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="list"] a {
        color: var(--text-color) !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] button {
        color: var(--text-color) !important;
        background-color: var(--card-bg) !important;
        border: 1px solid var(--accent-color) !important;
    }
    
    [data-testid="stSidebar"] button:hover {
        background-color: var(--accent-color) !important;
        color: var(--main-bg) !important;
    }
    
    /* Sidebar info boxes - Card background */
    [data-testid="stSidebar"] .stAlert,
    [data-testid="stSidebar"] .stInfo {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--accent-color);
    }
    
    /* Sidebar markdown text */
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-color) !important;
    }
    
    /* Sidebar radio buttons and selectboxes */
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        color: var(--text-color) !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="select"] label {
        color: var(--text-color) !important;
    }
    
    /* Main content text - White */
    .main h1,
    .main h2,
    .main h3,
    .main h4,
    .main p,
    .main div,
    .main span,
    .main label,
    .main strong {
        color: var(--text-color) !important;
    }
    
    /* Header styling - Accent color */
    h1 {
        color: var(--accent-color) !important;
        border-bottom: 3px solid var(--accent-color);
        padding-bottom: 10px;
    }
    
    h2 {
        color: var(--accent-color) !important;
        margin-top: 30px;
    }
    
    h3 {
        color: var(--accent-color) !important;
    }
    
    h4 {
        color: var(--text-color) !important;
    }
    
    /* Metric cards - Card background */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: var(--accent-color) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: var(--text-color) !important;
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background-color: var(--card-bg) !important;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--accent-color);
    }
    
    /* Button styling - Accent color */
    .stButton>button {
        background-color: var(--accent-color) !important;
        color: var(--main-bg) !important;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #0EA5E9 !important;
        color: var(--text-color) !important;
    }
    
    /* Info boxes - Card background */
    .stAlert,
    .stInfo,
    .stSuccess,
    .stWarning,
    .stError {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--accent-color) !important;
    }
    
    /* Dataframe styling - Card background */
    .dataframe {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
    }
    
    /* Table styling */
    table {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
    }
    
    table th {
        background-color: var(--sidebar-bg) !important;
        color: var(--accent-color) !important;
    }
    
    table td {
        color: var(--text-color) !important;
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--accent-color) !important;
    }
    
    /* Radio buttons and checkboxes */
    [data-baseweb="radio"] label,
    [data-baseweb="checkbox"] label {
        color: var(--text-color) !important;
    }
    
    /* Selectbox */
    [data-baseweb="select"] {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
    }
    
    /* Slider */
    [data-baseweb="slider"] {
        color: var(--accent-color) !important;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom containers - Card background */
    .metric-container {
        background-color: var(--card-bg) !important;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(56, 189, 248, 0.2);
        margin: 0.5rem 0;
        border: 1px solid var(--accent-color);
        color: var(--text-color) !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: var(--text-color) !important;
    }
    
    /* Caption text */
    .stCaption {
        color: var(--text-color) !important;
        opacity: 0.8;
    }
    
    /* Code blocks */
    code {
        background-color: var(--card-bg) !important;
        color: var(--accent-color) !important;
    }
    
    pre {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--accent-color) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main page content (Overview)
st.title("🏪 Mall Movement Tracking Dashboard")
st.markdown("---")

# Welcome message
st.info("""
# Welcome to Mall Movement Tracking Dashboard

This dashboard provides comprehensive ML-powered analytics for customer movement patterns in shopping malls.

**Available Pages (use sidebar navigation):**
- 📊 Overview - Dashboard home and key metrics
- 🔍 Data Explorer - Interactive data exploration
- 🗺️ Heatmaps - Movement pattern visualizations
- 🎯 Classification Results - Model performance metrics
- 👥 Clustering Insights - Customer segmentation analysis
- 📈 Forecasting Traffic - Traffic prediction models
- 🔮 Predict Next Zone - Real-time predictions
- 🧠 Model Explainability - Feature importance and model insights
""")

# Sidebar
st.sidebar.title("🏪 Navigation")
st.sidebar.markdown("---")
st.sidebar.info("""
**Mall Movement Tracking**

ML-powered analytics dashboard

**Version:** 1.0.0
""")

