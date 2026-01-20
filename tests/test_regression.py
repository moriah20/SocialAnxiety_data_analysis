import pytest
import pandas as pd
import numpy as np
import math
from Regression.regression import calculate_age_regression, extract_correlation_from_regression, interpret_regression_significance

@pytest.fixture
def regression_data():
    """Generates a synthetic dataset with a known positive correlation."""
    np.random.seed(42)
    age = np.linspace(20, 60, 50)
    # Target = Age * 2 + some noise
    target = age * 2 + np.random.normal(0, 5, 50)
    return pd.DataFrame({'Age': age, 'Score': target})

@pytest.fixture
def dirty_data():
    """Generates data with strings and NaNs to test cleaning logic."""
    return pd.DataFrame({
        'Age': [25, 30, 'invalid', 40, np.nan],
        'Score': [100, 110, 120, 'wrong', 140]
    })

# --- Tests for calculate_age_regression ---

def test_regression_success(regression_data):
    """Checks if the regression returns a valid Pingouin DataFrame."""
    results = calculate_age_regression(regression_data, "Score", "Age")
    assert isinstance(results, pd.DataFrame)
    assert 'coef' in results.columns
    assert 'p-val' in results.columns

def test_regression_cleaning(dirty_data):
    """Checks if the function handles non-numeric data and NaNs correctly."""
    # After cleaning, only rows 0 and 1 should remain
    results = calculate_age_regression(dirty_data, "Score", "Age")
    assert results is not None
    # Check if the coefficient is calculated based on the valid rows
    assert not np.isnan(results.iloc[1]['coef'])

# --- Tests for extract_correlation_from_regression ---

def test_correlation_direction_and_strength(regression_data):
    """Checks if the correlation magnitude and direction are correctly interpreted."""
    results = calculate_age_regression(regression_data, "Score", "Age")
    corr = extract_correlation_from_regression(results)
    
    assert corr > 0  # Should be positive
    assert 0.7 <= abs(corr) <= 1.0  # Should be strong based on our dummy data

def test_correlation_handling_errors():
    """Checks if the function returns None when passed invalid results."""
    assert extract_correlation_from_regression(None) is None

# --- Tests for interpret_regression_significance ---

def test_significance_logic(regression_data):
    """Checks if p-value extraction works."""
    results = calculate_age_regression(regression_data, "Score", "Age")
    p_val = interpret_regression_significance(results)
    
    assert isinstance(p_val, float)
    assert 0 <= p_val <= 1