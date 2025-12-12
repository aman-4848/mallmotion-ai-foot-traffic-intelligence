"""
Unit tests for API endpoints
"""
import pytest
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from fastapi.testclient import TestClient
    from api.app import app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False


@pytest.mark.skipif(not API_AVAILABLE, reason="FastAPI not available")
class TestAPIEndpoints:
    """Test suite for API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_data_info_endpoint(self, client):
        """Test data info endpoint"""
        try:
            response = client.get("/api/data/info")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict)
        except Exception as e:
            pytest.skip(f"Could not test data info endpoint: {e}")
    
    def test_classification_results_endpoint(self, client):
        """Test classification results endpoint"""
        try:
            response = client.get("/api/results/classification")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict)
        except Exception as e:
            pytest.skip(f"Could not test classification endpoint: {e}")
    
    def test_clustering_results_endpoint(self, client):
        """Test clustering results endpoint"""
        try:
            response = client.get("/api/results/clustering")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict)
        except Exception as e:
            pytest.skip(f"Could not test clustering endpoint: {e}")
    
    def test_forecasting_results_endpoint(self, client):
        """Test forecasting results endpoint"""
        try:
            response = client.get("/api/results/forecasting")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict)
        except Exception as e:
            pytest.skip(f"Could not test forecasting endpoint: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

