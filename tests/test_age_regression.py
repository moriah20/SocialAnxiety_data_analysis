import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from Age_Regression.regression import extract_correlation_from_regression, interpret_regression, plot_regression,calculate_regression

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
    """Generates data with enough valid numeric rows after cleaning."""
    return pd.DataFrame({
        'Age': [25, 30, 'invalid', 40, np.nan, 50, 55, 60], 
        'Score': [100, 110, 120, 'wrong', 140, 150, 160, 170] 
    })

# --- Tests for calculate_age_regression ---

# --- Tests for extract_correlation_from_regression ---

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

# --- Additional Tests for Logic ---

def test_calculate_regression_success(regression_data):
    """Verifies that regression returns a valid DataFrame with results."""
    results = calculate_regression(regression_data, 'Score', 'Age')
    assert isinstance(results, pd.DataFrame)
    assert 'pval' in results.columns
    assert results.iloc[1]['pval'] < 0.05  # Should be significant for our synthetic data

def test_calculate_regression_cleaning(dirty_data):
    """Verifies that the function cleans strings and NaNs before regression."""
    # This should not crash and return a valid result based on numeric rows
    results = calculate_regression(dirty_data, 'Score', 'Age')
    assert results is not None

def test_extract_correlation_logic(regression_data):
    """Checks if correlation magnitude and direction are correctly extracted."""
    results = calculate_regression(regression_data, 'Score', 'Age')
    corr = extract_correlation_from_regression(results)
    # Our synthetic data has a strong positive correlation
    assert corr > 0.7 
    assert isinstance(corr, float)

def test_interpret_regression_boolean(regression_data):
    results = calculate_regression(regression_data, 'Score', 'Age')
    # Use == instead of 'is' to avoid Numpy type conflicts
    assert interpret_regression(results) == True

# --- Smoke Test for Plotting ---

def test_plot_regression_no_crash(regression_data, tmp_path):
    try:
        plot_regression(
            regression_data,
            'Age',
            'Score',
            output_dir=tmp_path
        )

        # Optional: verify that a file was saved
        saved = list(tmp_path.glob("*.png"))
        assert len(saved) > 0

        plt.close('all')

    except Exception as e:
        pytest.fail(f"Plotting crashed: {e}")


