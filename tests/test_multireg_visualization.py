import pytest
import pandas as pd
import logging
from unittest.mock import patch, MagicMock

from Statistic_Analysis_for_Binaraziation.stat_visualization import plot_regression_summary

# --- Fixtures ---

@pytest.fixture
def mock_model():
    """
    Creates a mock statsmodels OLS result object.
    It simulates the attributes required by the plotting function:
    - .params (coefficients)
    - .conf_int() (confidence intervals)
    """
    mock = MagicMock()
    
    # 1. Simulate Coefficients (params)
    # We include 'const' to verify it gets removed, and other variables to be plotted.
    mock.params = pd.Series(
        {'const': 0.1, 'Lifestyle': 0.5, 'Health': -0.3, 'History': 0.2}
    )
    
    # 2. Simulate Confidence Intervals (conf_int)
    # Returns a DataFrame with column 0 (lower bound) and column 1 (upper bound)
    mock.conf_int.return_value = pd.DataFrame({
        0: [0.0, 0.4, -0.5, 0.1],  # Lower CI
        1: [0.2, 0.6, -0.1, 0.3]   # Upper CI
    }, index=['const', 'Lifestyle', 'Health', 'History'])
    
    return mock

# --- Tests ---

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.errorbar')
def test_plot_regression_success(mock_errorbar, mock_show, mock_model, caplog):
    """
    Test the 'Happy Path':
    1. Ensures the function runs without error using a valid model.
    2. Verifies that 'errorbar' was called (graph created).
    3. Verifies that 'const' (Intercept) was removed from the plot data.
    """
    # 1. Run the function
    with caplog.at_level(logging.INFO):
        plot_regression_summary(mock_model)
        
    # 2. Verify that plotting functions were called
    mock_show.assert_called_once()
    mock_errorbar.assert_called_once()
    
    # 3. Verify Data Processing (Crucial Step)
    # We inspect the arguments passed to plt.errorbar to ensure 'const' is gone.
    # args[0] is x (index), args[1] is y (coef)
    call_args = mock_errorbar.call_args
    x_values = call_args.kwargs.get('x') if 'x' in call_args.kwargs else call_args[0][0]
    
    # Check that 'const' is NOT in the x-axis labels
    assert 'const' not in x_values, "The Intercept (const) should be removed from the plot."
    assert 'Lifestyle' in x_values
    
    # 4. Verify Logging
    assert "Generating Regression Coefficients plot..." in caplog.text
    assert "Intercept (const) removed from visualization data" in caplog.text
    assert "Displaying Regression plot window." in caplog.text


@patch('matplotlib.pyplot.show')
def test_plot_regression_no_const(mock_show, mock_model, caplog):
    """
    Test scenario where the model has NO constant/intercept.
    The function should still run correctly without crashing.
    """
    # Remove 'const' from the mock data
    mock_model.params = mock_model.params.drop('const')
    mock_model.conf_int.return_value = mock_model.conf_int.return_value.drop('const')
    
    # Run function
    plot_regression_summary(mock_model)
    
    # Verify execution
    mock_show.assert_called_once()
    # Ensure we didn't log the removal message since there was nothing to remove
    assert "Intercept (const) removed" not in caplog.text


@patch('matplotlib.pyplot.errorbar')
def test_plot_regression_error_handling(mock_errorbar, mock_model, caplog):
    """
    Test Exception Handling:
    Simulate a plotting error to ensure the function catches it, logs it, and re-raises it.
    """
    # Force the plotting library to crash
    mock_errorbar.side_effect = Exception("Matplotlib Error")
    
    # Expect the function to raise the exception
    with pytest.raises(Exception, match="Matplotlib Error"):
        plot_regression_summary(mock_model)
        
    # Verify the error was logged
    assert "An error occurred while plotting the Regression summary" in caplog.text
    assert any(record.levelname == "ERROR" for record in caplog.records)