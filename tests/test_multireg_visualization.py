import pytest
import pandas as pd
import logging
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

# Import the functions from your module
# Replace 'your_module_name' with the actual name of your python file
from category_realations.stat_visualization import plot_spearman_bar_chart, plot_regression_summary

# --- Fixtures ---

@pytest.fixture
def mock_spearman_data():
    """Provides a valid DataFrame for Spearman correlation plots."""
    return pd.DataFrame({
        'Category': ['Lifestyle', 'Health', 'History'],
        'Spearman Coeff': [0.45, -0.12, 0.88]
    })

@pytest.fixture
def mock_regression_model():
    """
    Creates a mock statsmodels OLS result object.
    Includes 'const' to verify it gets removed during plotting.
    """
    mock = MagicMock()
    indices = ['const', 'Lifestyle', 'Health', 'History']
    
    # Simulate coefficients (params)
    mock.params = pd.Series([0.1, 0.5, -0.3, 0.2], index=indices)
    
    # Simulate confidence intervals (conf_int)
    mock.conf_int.return_value = pd.DataFrame({
        0: [0.0, 0.4, -0.5, 0.1],  # Lower CI
        1: [0.2, 0.6, -0.1, 0.3]   # Upper CI
    }, index=indices)
    
    return mock

# --- Spearman Chart Tests ---

@patch('matplotlib.pyplot.show')
def test_plot_spearman_success(mock_show, mock_spearman_data, caplog):
    """Verifies that the Spearman bar chart is generated correctly with valid data."""
    with caplog.at_level(logging.INFO):
        plot_spearman_bar_chart(mock_spearman_data)
    
    assert "Generating Spearman correlation bar chart..." in caplog.text
    assert "Displaying Spearman plot window." in caplog.text
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.show')
def test_plot_spearman_empty_df(mock_show, caplog):
    """Verifies that the function skips plotting and logs a warning for empty DataFrames."""
    empty_df = pd.DataFrame(columns=['Category', 'Spearman Coeff'])
    
    with caplog.at_level(logging.WARNING):
        plot_spearman_bar_chart(empty_df)
    
    assert "Spearman DataFrame is empty. Skipping plot." in caplog.text
    mock_show.assert_not_called()

# --- Regression Summary Tests ---

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.errorbar')
def test_plot_regression_success(mock_errorbar, mock_show, mock_regression_model, caplog):
    """
    Tests the happy path for regression visualization.
    Checks for:
    1. Successful execution and logging.
    2. Removal of the Intercept (const).
    3. Proper data conversion to avoid Matplotlib errors.
    """
    with caplog.at_level(logging.INFO):
        plot_regression_summary(mock_regression_model)
        
    # Verify plotting functions were triggered
    mock_show.assert_called_once()
    mock_errorbar.assert_called_once()
    
    # Extract arguments passed to plt.errorbar
    call_args = mock_errorbar.call_args
    x_values = call_args.kwargs.get('x') if 'x' in call_args.kwargs else call_args[0][0]
    
    # FIXED: Convert x_values to a list to ensure compatibility with our manual fix
    # and verify that 'const' was successfully removed.
    x_values_list = list(x_values)
    
    assert 'const' not in x_values_list, "The Intercept (const) should be removed from the plot data."
    assert 'Lifestyle' in x_values_list
    assert "Generating Regression Coefficients plot..." in caplog.text

@patch('matplotlib.pyplot.show')
def test_plot_regression_no_const(mock_show, mock_regression_model, caplog):
    """Verifies the function works even if the model has no intercept ('const')."""
    # Remove 'const' from the mock data
    mock_regression_model.params = mock_regression_model.params.drop('const')
    mock_regression_model.conf_int.return_value = mock_regression_model.conf_int.return_value.drop('const')
    
    plot_regression_summary(mock_regression_model)
    
    mock_show.assert_called_once()
    # Check that debug log for removal didn't trigger
    assert "Intercept (const) removed" not in caplog.text

@patch('matplotlib.pyplot.figure')
def test_plot_regression_error_handling(mock_figure, mock_regression_model, caplog):
    """Simulates a Matplotlib crash to verify exception handling and logging."""
    mock_figure.side_effect = Exception("Matplotlib failure")
    
    with pytest.raises(Exception, match="Matplotlib failure"):
        plot_regression_summary(mock_regression_model)
        
    assert "An error occurred while plotting the Regression summary" in caplog.text