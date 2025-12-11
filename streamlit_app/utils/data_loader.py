"""
Data Loader Utility
Loads processed/cleaned data from the processed folder
"""
import pandas as pd
import os
from pathlib import Path

# Get project root (assuming this file is in streamlit_app/utils)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_FILE = DATA_DIR / "merged data set.csv"

def load_processed_data(file_path=None):
    """
    Load the processed/cleaned merged dataset
    
    Args:
        file_path: Optional path to data file. Defaults to processed/merged data set.csv
    
    Returns:
        pandas DataFrame
    """
    if file_path is None:
        file_path = PROCESSED_DATA_FILE
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data file not found: {file_path}")
    
    # Try CSV first, then Excel
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return df

def get_data_info(df):
    """Get basic information about the dataset"""
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }

