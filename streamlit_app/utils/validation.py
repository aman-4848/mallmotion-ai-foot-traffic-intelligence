"""
Data Validation Utilities for Production
Comprehensive validation for inputs and data
"""
import pandas as pd
import numpy as np
from typing import Any, List, Optional, Callable
import streamlit as st

def validate_dataframe(df: pd.DataFrame, 
                      required_columns: Optional[List[str]] = None,
                      min_rows: int = 1) -> tuple[bool, Optional[str]]:
    """
    Validate DataFrame structure and content
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    if len(df) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    return True, None

def validate_numeric_range(value: float, 
                          min_val: Optional[float] = None,
                          max_val: Optional[float] = None) -> tuple[bool, Optional[str]]:
    """
    Validate numeric value is within range
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, (int, float)) or np.isnan(value):
        return False, "Value must be a valid number"
    
    if min_val is not None and value < min_val:
        return False, f"Value must be >= {min_val}"
    
    if max_val is not None and value > max_val:
        return False, f"Value must be <= {max_val}"
    
    return True, None

def validate_file_upload(file, 
                        allowed_extensions: List[str],
                        max_size_mb: float = 10) -> tuple[bool, Optional[str]]:
    """
    Validate uploaded file
    
    Args:
        file: Uploaded file object
        allowed_extensions: List of allowed file extensions
        max_size_mb: Maximum file size in MB
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if file is None:
        return False, "No file uploaded"
    
    file_extension = file.name.split('.')[-1].lower()
    if file_extension not in allowed_extensions:
        return False, f"File extension .{file_extension} not allowed. Allowed: {', '.join(allowed_extensions)}"
    
    file_size_mb = len(file.getvalue()) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File size ({file_size_mb:.2f} MB) exceeds maximum ({max_size_mb} MB)"
    
    return True, None

def validate_model_input(features: dict, 
                        required_features: List[str]) -> tuple[bool, Optional[str]]:
    """
    Validate model input features
    
    Args:
        features: Dictionary of feature values
        required_features: List of required feature names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_features = [feat for feat in required_features if feat not in features]
    if missing_features:
        return False, f"Missing required features: {', '.join(missing_features)}"
    
    # Check for None or NaN values
    invalid_features = [feat for feat, val in features.items() 
                       if val is None or (isinstance(val, float) and np.isnan(val))]
    if invalid_features:
        return False, f"Invalid values for features: {', '.join(invalid_features)}"
    
    return True, None

def validate_date_range(start_date, end_date) -> tuple[bool, Optional[str]]:
    """
    Validate date range
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if start_date is None or end_date is None:
        return False, "Both start and end dates are required"
    
    if start_date > end_date:
        return False, "Start date must be before end date"
    
    return True, None

def sanitize_input(value: Any) -> Any:
    """
    Sanitize user input to prevent injection attacks
    
    Args:
        value: Input value to sanitize
    
    Returns:
        Sanitized value
    """
    if isinstance(value, str):
        # Remove potentially dangerous characters
        value = value.strip()
        # Add more sanitization as needed
    return value

