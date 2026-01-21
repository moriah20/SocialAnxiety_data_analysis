import pytest
import pandas as pd
import numpy as np
import math
from Regression.regression import calculate_age_regression, extract_correlation_from_regression, interpret_regression

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
def interpret_regression_significance(results):
    """
    Extracts the p-value of the primary predictor from regression results.
    """
    # Check if results exist and have at least the Intercept and one Predictor
    if results is None or not isinstance(results, pd.DataFrame) or len(results) < 2:
        return None

    try:
        # results.iloc[1] is the row for the Independent Variable (e.g., Age)
        # 'pval' is the standard column name in pingouin.linear_regression
        p_value = results.iloc[1]['pval']
        return float(p_value)
    except (KeyError, IndexError, ValueError):
        return None


def test_regression_success(regression_data):
    """Checks if the regression returns a valid Pingouin DataFrame."""
    results = calculate_age_regression(regression_data, "Score", "Age")
    assert isinstance(results, pd.DataFrame)
    assert 'coef' in results.columns
    # FIXED: Pingouin uses 'pval', not 'p-val'
    assert 'pval' in results.columns

def test_regression_cleaning(dirty_data):
    """Checks if the function handles non-numeric data and NaNs correctly."""
    # Adding more rows to ensure we have at least 3 valid samples after cleaning
    more_data = pd.DataFrame({
        'Age': [45.0, 50.0, 55.0],
        'Score': [130.0, 140.0, 150.0]
    })
    extended_dirty_data = pd.concat([dirty_data, more_data], ignore_index=True)
    
    results = calculate_age_regression(extended_dirty_data, "Score", "Age")
    assert results is not None
    assert isinstance(results, pd.DataFrame)

