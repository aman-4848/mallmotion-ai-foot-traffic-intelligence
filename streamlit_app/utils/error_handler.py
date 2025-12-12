"""
Error Handling Utilities for Production
Comprehensive error handling and user-friendly error messages
"""
import streamlit as st
import traceback
import logging
from pathlib import Path
from typing import Optional, Callable, Any
import functools

# Setup logging
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AppError(Exception):
    """Custom application error"""
    def __init__(self, message: str, user_message: Optional[str] = None):
        self.message = message
        self.user_message = user_message or "An error occurred. Please try again."
        super().__init__(self.message)

def handle_errors(func: Callable) -> Callable:
    """
    Decorator to handle errors gracefully
    
    Args:
        func: Function to wrap
    
    Returns:
        Wrapped function with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            st.error(f"❌ File not found: {str(e)}")
            st.info("💡 Please check that the file exists and the path is correct.")
            return None
        except ValueError as e:
            logger.error(f"Value error: {e}")
            st.error(f"❌ Invalid value: {str(e)}")
            st.info("💡 Please check your input values and try again.")
            return None
        except KeyError as e:
            logger.error(f"Key error: {e}")
            st.error(f"❌ Missing required data: {str(e)}")
            st.info("💡 The requested data field is not available.")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}\n{traceback.format_exc()}")
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            st.info("💡 Please try again or contact support if the problem persists.")
            with st.expander("🔍 Technical Details"):
                st.code(traceback.format_exc())
            return None
    return wrapper

def safe_execute(func: Callable, *args, **kwargs) -> tuple[Any, Optional[str]]:
    """
    Safely execute a function and return result with error message
    
    Args:
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Tuple of (result, error_message)
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in {func.__name__}: {error_msg}")
        return None, error_msg

def show_error(error: Exception, context: str = ""):
    """
    Display user-friendly error message
    
    Args:
        error: Exception object
        context: Additional context about where error occurred
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    logger.error(f"{context}: {error_type} - {error_msg}")
    
    st.error(f"❌ **{error_type}**")
    st.error(f"{error_msg}")
    
    if context:
        st.info(f"📍 Context: {context}")
    
    # Show technical details in expander
    with st.expander("🔍 Technical Details"):
        st.code(traceback.format_exc())

def validate_input(value: Any, validation_func: Callable, error_msg: str) -> bool:
    """
    Validate input and show error if invalid
    
    Args:
        value: Value to validate
        validation_func: Function that returns True if valid
        error_msg: Error message to show if invalid
    
    Returns:
        True if valid, False otherwise
    """
    if not validation_func(value):
        st.error(f"❌ {error_msg}")
        return False
    return True

def log_operation(operation: str):
    """
    Decorator to log operations
    
    Args:
        operation: Name of the operation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Starting {operation}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed {operation}")
                return result
            except Exception as e:
                logger.error(f"Failed {operation}: {e}")
                raise
        return wrapper
    return decorator

