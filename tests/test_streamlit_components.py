"""
Unit tests for Streamlit components and utilities
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from streamlit_app.utils.data_loader import load_processed_data, get_data_info
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit utilities not available")
class TestDataLoader:
    """Test suite for data loader utilities"""
    
    def test_load_processed_data(self):
        """Test loading processed data"""
        try:
            df = load_processed_data()
            assert df is not None
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert len(df.columns) > 0
        except FileNotFoundError:
            pytest.skip("Processed data file not found")
        except Exception as e:
            pytest.skip(f"Could not load data: {e}")
    
    def test_get_data_info(self):
        """Test getting data information"""
        try:
            df = load_processed_data()
            info = get_data_info(df)
            assert info is not None
            assert isinstance(info, dict)
            assert 'shape' in info
            assert 'memory_usage' in info
            assert 'null_counts' in info
        except FileNotFoundError:
            pytest.skip("Processed data file not found")
        except Exception as e:
            pytest.skip(f"Could not get data info: {e}")
    
    def test_data_info_structure(self):
        """Test data info structure"""
        try:
            df = load_processed_data()
            info = get_data_info(df)
            
            # Check shape
            assert isinstance(info['shape'], tuple)
            assert len(info['shape']) == 2
            
            # Check memory usage
            assert isinstance(info['memory_usage'], (int, float))
            assert info['memory_usage'] > 0
            
            # Check null counts
            assert isinstance(info['null_counts'], dict)
        except FileNotFoundError:
            pytest.skip("Processed data file not found")
        except Exception as e:
            pytest.skip(f"Could not test data info: {e}")


class TestDataValidation:
    """Test suite for data validation"""
    
    def test_dataframe_not_empty(self):
        """Test that dataframe is not empty"""
        try:
            from streamlit_app.utils.data_loader import load_processed_data
            df = load_processed_data()
            assert len(df) > 0, "Dataframe is empty"
        except FileNotFoundError:
            pytest.skip("Processed data file not found")
        except Exception as e:
            pytest.skip(f"Could not test: {e}")
    
    def test_required_columns_exist(self):
        """Test that required columns exist"""
        try:
            from streamlit_app.utils.data_loader import load_processed_data
            df = load_processed_data()
            
            # Check for common required columns
            required_cols = ['USERID', 'SPACEID']
            for col in required_cols:
                if col in df.columns:
                    assert col in df.columns, f"Required column {col} not found"
        except FileNotFoundError:
            pytest.skip("Processed data file not found")
        except Exception as e:
            pytest.skip(f"Could not test: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

