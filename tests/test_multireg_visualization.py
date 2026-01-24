import pytest
import pandas as pd
import logging
from unittest.mock import MagicMock, patch

from Category_realations.stat_visualization import plot_regression_summary

# --- Fixtures ---

@pytest.fixture
def mock_regression_model():
    """
    Creates a mock object that mimics a Statsmodels OLS result.
    It simulates '.params' and '.conf_int()'.
    """
    # 1. Create a generic Mock object
    mock_model = MagicMock()
    
    # 2. Mock 'params' (The coefficients)
    # We include 'const' to verify it gets dropped later
    mock_model.params = pd.Series(
        data=[5.0, 2.5, -1.2], 
        index=['const', 'Lifestyle', 'Health']
    )
    
    # 3. Mock 'conf_int()' (Confidence Intervals)
    # Returns a DataFrame with lower (0) and upper (1) bounds
    conf_int_df = pd.DataFrame(
        data={
            0: [4.0, 2.0, -1.5], # Lower CI
            1: [6.0, 3.0, -0.9]  # Upper CI
        },
        index=['const', 'Lifestyle', 'Health']
    )
    mock_model.conf_int.return_value = conf_int_df
    
    return mock_model

# --- Tests ---

@patch("Category_realations.stat_visualization.plt")
def test_plot_regression_summary_success(mock_plt, mock_regression_model):
    """
    Verifies that the regression summary plot is generated correctly.
    Checks specifically if the 'const' (intercept) is dropped.
    """
    # Run the function
    plot_regression_summary(mock_regression_model)
    
    # 1. Assert figure creation
    mock_plt.figure.assert_called_once()
    
    # 2. Assert errorbar plot was called
    mock_plt.errorbar.assert_called_once()
    
    # 3. CRITICAL: Verify that 'const' was removed from the data passed to plot
    # Access the arguments passed to plt.errorbar
    args, kwargs = mock_plt.errorbar.call_args
    
    # 'x' argument is usually the first positional arg or in kwargs
    x_data = kwargs.get('x') if 'x' in kwargs else args[0]
    
    # Verify 'const' is NOT in the x-axis labels
    assert 'const' not in x_data, "The intercept 'const' should be removed from the visualization."
    assert 'Lifestyle' in x_data
    assert 'Health' in x_data

    # 4. Verify title setup
    mock_plt.title.assert_called_with('Regression Coefficients: Impact on Anxiety Level', fontsize=14, fontweight='bold')