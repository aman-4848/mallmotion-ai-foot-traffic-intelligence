"""
Preprocessing Utilities
Data preprocessing functions for Streamlit
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from features.feature_engineering import FeatureEngineer

def prepare_features_for_prediction(df, feature_columns):
    """
    Prepare features for model prediction
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature column names
    
    Returns:
        Feature matrix
    """
    # Select and fill features
    X = df[feature_columns].fillna(0)
    return X.values

def get_feature_columns(df, exclude_cols=None):
    """
    Get feature columns from DataFrame
    
    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude
    
    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = ['SPACEID', 'USERID']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    return feature_cols

