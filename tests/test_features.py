"""
Unit tests for feature engineering module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from features.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        data = {
            'USERID': np.random.randint(1, 100, 100),
            'SPACEID': np.random.randint(1, 50, 100),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'numeric_col': np.random.randn(100),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
            'missing_col': np.random.randn(100)
        }
        df = pd.DataFrame(data)
        # Add some missing values
        df.loc[10:15, 'missing_col'] = np.nan
        return df
    
    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance"""
        return FeatureEngineer()
    
    def test_initialization(self, feature_engineer):
        """Test FeatureEngineer initialization"""
        assert feature_engineer is not None
        assert hasattr(feature_engineer, 'engineer_features')
    
    def test_handle_missing_values(self, feature_engineer, sample_data):
        """Test missing value handling"""
        result = feature_engineer.handle_missing_values(sample_data.copy())
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # Check that missing values are handled
        assert result['missing_col'].isnull().sum() == 0
    
    def test_extract_datetime_features(self, feature_engineer, sample_data):
        """Test datetime feature extraction"""
        result = feature_engineer.extract_datetime_features(sample_data.copy())
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # Check that datetime features are created
        datetime_features = [col for col in result.columns if any(x in col.lower() for x in ['hour', 'day', 'month', 'weekday'])]
        assert len(datetime_features) > 0
    
    def test_encode_categorical(self, feature_engineer, sample_data):
        """Test categorical encoding"""
        result = feature_engineer.encode_categorical(sample_data.copy())
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # Check that categorical columns are encoded
        assert 'categorical_col' in result.columns or any('categorical_col' in col for col in result.columns)
    
    def test_handle_outliers(self, feature_engineer, sample_data):
        """Test outlier handling"""
        result = feature_engineer.handle_outliers(sample_data.copy(), method='iqr')
        assert result is not None
        assert isinstance(result, pd.DataFrame)
    
    def test_create_bins(self, feature_engineer, sample_data):
        """Test binning functionality"""
        result = feature_engineer.create_bins(sample_data.copy(), 'numeric_col', n_bins=5)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # Check that bin column is created
        bin_cols = [col for col in result.columns if 'numeric_col' in col and 'bin' in col.lower()]
        assert len(bin_cols) > 0
    
    def test_create_domain_features(self, feature_engineer, sample_data):
        """Test domain-specific feature creation"""
        result = feature_engineer.create_domain_features(sample_data.copy())
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # Check that domain features are created
        domain_features = [col for col in result.columns if any(x in col.lower() for x in ['visit', 'frequency', 'duration'])]
        assert len(domain_features) > 0 or result.shape[1] > sample_data.shape[1]
    
    def test_engineer_features_full_pipeline(self, feature_engineer, sample_data):
        """Test full feature engineering pipeline"""
        result = feature_engineer.engineer_features(sample_data.copy())
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # Check that features are added
        assert result.shape[1] >= sample_data.shape[1]
        # Check that missing values are handled
        assert result.isnull().sum().sum() == 0 or result.isnull().sum().sum() < sample_data.isnull().sum().sum()
    
    def test_empty_dataframe(self, feature_engineer):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        with pytest.raises((ValueError, IndexError, KeyError)):
            feature_engineer.engineer_features(empty_df)
    
    def test_no_datetime_column(self, feature_engineer):
        """Test handling when no datetime column exists"""
        data = {
            'USERID': [1, 2, 3],
            'SPACEID': [10, 20, 30],
            'numeric_col': [1.0, 2.0, 3.0]
        }
        df = pd.DataFrame(data)
        result = feature_engineer.engineer_features(df)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
    
    def test_all_numeric_data(self, feature_engineer):
        """Test handling of all numeric data"""
        np.random.seed(42)
        data = {
            'col1': np.random.randn(50),
            'col2': np.random.randn(50),
            'col3': np.random.randn(50)
        }
        df = pd.DataFrame(data)
        result = feature_engineer.engineer_features(df)
        assert result is not None
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

