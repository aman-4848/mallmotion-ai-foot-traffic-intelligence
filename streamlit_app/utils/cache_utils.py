"""
Caching Utilities for Production
Implements caching strategies for improved performance
"""
import streamlit as st
from functools import wraps
import hashlib
import pickle
from pathlib import Path
import pandas as pd
import joblib
from typing import Any, Callable

# Cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

def cache_data(ttl: int = 3600, show_spinner: bool = True):
    """
    Enhanced caching decorator with TTL support
    
    Args:
        ttl: Time to live in seconds (default: 1 hour)
        show_spinner: Whether to show loading spinner
    """
    def decorator(func: Callable) -> Callable:
        @st.cache_data(ttl=ttl, show_spinner=show_spinner)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def cache_model(model_path: str):
    """
    Cache model loading to avoid reloading
    
    Args:
        model_path: Path to model file
    """
    @st.cache_resource
    def load_model(path: str):
        return joblib.load(path)
    
    return load_model(model_path)

def get_cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key from function arguments
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Cache key string
    """
    key_data = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_data.encode()).hexdigest()

def clear_cache():
    """Clear all cached data"""
    st.cache_data.clear()
    st.cache_resource.clear()

def get_cache_stats() -> dict:
    """
    Get cache statistics
    
    Returns:
        Dictionary with cache statistics
    """
    # This would need to be implemented based on Streamlit's cache API
    return {
        "cache_enabled": True,
        "cache_dir": str(CACHE_DIR)
    }

