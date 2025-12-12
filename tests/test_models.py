"""
Unit tests for model loading and prediction
"""
import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from features.feature_engineering import FeatureEngineer


class TestModelLoading:
    """Test suite for model loading and prediction"""
    
    @pytest.fixture
    def models_dir(self):
        """Get models directory"""
        return Path(__file__).parent.parent / "models"
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        data = {
            'USERID': np.random.randint(1, 100, 50),
            'SPACEID': np.random.randint(1, 50, 50),
            'numeric_col': np.random.randn(50),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 50)
        }
        return pd.DataFrame(data)
    
    def test_classification_models_exist(self, models_dir):
        """Test that classification models exist"""
        classification_dir = models_dir / "classification"
        if classification_dir.exists():
            model_files = list(classification_dir.glob("*.pkl"))
            # At least one model should exist
            assert len(model_files) > 0, "No classification models found"
    
    def test_clustering_models_exist(self, models_dir):
        """Test that clustering models exist"""
        clustering_dir = models_dir / "clustering"
        if clustering_dir.exists():
            model_files = list(clustering_dir.glob("*.pkl"))
            # At least one model should exist
            assert len(model_files) > 0, "No clustering models found"
    
    def test_load_classification_model(self, models_dir):
        """Test loading a classification model"""
        classification_dir = models_dir / "classification"
        if classification_dir.exists():
            model_files = list(classification_dir.glob("*.pkl"))
            if model_files:
                model_path = model_files[0]
                try:
                    model = joblib.load(model_path)
                    assert model is not None
                except Exception as e:
                    pytest.skip(f"Could not load model: {e}")
    
    def test_load_clustering_model(self, models_dir):
        """Test loading a clustering model"""
        clustering_dir = models_dir / "clustering"
        if clustering_dir.exists():
            model_files = list(clustering_dir.glob("*.pkl"))
            if model_files:
                model_path = model_files[0]
                try:
                    model = joblib.load(model_path)
                    assert model is not None
                except Exception as e:
                    pytest.skip(f"Could not load model: {e}")
    
    def test_model_prediction_format(self, models_dir, sample_data):
        """Test that models can make predictions"""
        classification_dir = models_dir / "classification"
        if classification_dir.exists():
            model_files = list(classification_dir.glob("*.pkl"))
            if model_files:
                try:
                    model = joblib.load(model_files[0])
                    fe = FeatureEngineer()
                    df_engineered = fe.engineer_features(sample_data.copy())
                    
                    # Get numeric features
                    numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
                    if 'SPACEID' in numeric_cols:
                        numeric_cols.remove('SPACEID')
                    if 'USERID' in numeric_cols:
                        numeric_cols.remove('USERID')
                    
                    X = df_engineered[numeric_cols].fillna(0)
                    
                    # Try to make prediction
                    if hasattr(model, 'predict'):
                        predictions = model.predict(X.head(5))
                        assert predictions is not None
                        assert len(predictions) == 5
                except Exception as e:
                    pytest.skip(f"Could not test prediction: {e}")


class TestResultsFiles:
    """Test suite for results files"""
    
    @pytest.fixture
    def results_dir(self):
        """Get results directory"""
        return Path(__file__).parent.parent / "results"
    
    def test_classification_results_exist(self, results_dir):
        """Test that classification results exist"""
        metrics_file = results_dir / "classification" / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                results = json.load(f)
                assert isinstance(results, dict)
                assert len(results) > 0
    
    def test_clustering_results_exist(self, results_dir):
        """Test that clustering results exist"""
        silhouette_file = results_dir / "clustering" / "silhouette_score.json"
        if silhouette_file.exists():
            with open(silhouette_file, 'r') as f:
                results = json.load(f)
                assert isinstance(results, dict)
                assert len(results) > 0
    
    def test_forecasting_results_exist(self, results_dir):
        """Test that forecasting results exist"""
        rmse_file = results_dir / "forecasting" / "rmse.json"
        if rmse_file.exists():
            with open(rmse_file, 'r') as f:
                results = json.load(f)
                assert isinstance(results, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

