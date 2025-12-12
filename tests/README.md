# Testing Guide

This directory contains comprehensive tests for the Mall Movement Tracking project.

## Test Structure

- `test_features.py` - Feature engineering tests
- `test_models.py` - Model loading and prediction tests
- `test_streamlit_components.py` - Streamlit utility tests
- `test_api.py` - API endpoint tests

## Running Tests

### Install Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_features.py -v
pytest tests/test_models.py -v
pytest tests/test_streamlit_components.py -v
pytest tests/test_api.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Categories

### 1. Feature Engineering Tests (`test_features.py`)
- FeatureEngineer initialization
- Missing value handling
- Datetime feature extraction
- Categorical encoding
- Outlier handling
- Binning functionality
- Domain feature creation
- Full pipeline testing

### 2. Model Tests (`test_models.py`)
- Model file existence
- Model loading
- Prediction format validation
- Results file validation

### 3. Streamlit Component Tests (`test_streamlit_components.py`)
- Data loading functionality
- Data info generation
- Data validation
- Required columns check

### 4. API Tests (`test_api.py`)
- Root endpoint
- Data info endpoint
- Classification results endpoint
- Clustering results endpoint
- Forecasting results endpoint

## Writing New Tests

### Example Test Structure

```python
import pytest
from your_module import YourClass

class TestYourClass:
    @pytest.fixture
    def sample_data(self):
        # Create test data
        return pd.DataFrame(...)
    
    def test_your_function(self, sample_data):
        # Test implementation
        result = YourClass().your_function(sample_data)
        assert result is not None
```

## Best Practices

1. **Use Fixtures**: Create reusable test data with fixtures
2. **Test Edge Cases**: Test empty data, missing columns, etc.
3. **Skip When Appropriate**: Use `pytest.skip()` for optional dependencies
4. **Clear Assertions**: Use descriptive assertion messages
5. **Isolated Tests**: Each test should be independent

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: pytest tests/ -v --cov
```

---

**Last Updated**: 2024-12-11

