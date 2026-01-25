import pytest
import pandas as pd
import logging
from unittest.mock import patch

from Category_realations.stat_visualization import plot_spearman_bar_chart

# --- Fixtures ---

@pytest.fixture
def mock_spearman_df():
    """
    Creates a dummy DataFrame representing Spearman correlation results.
    Used specifically for the Spearman bar chart tests.
    """
    data = {
        'Category': ['Lifestyle', 'Health', 'History'],
        'Spearman Coeff': [0.85, -0.4, 0.1]
    }
    return pd.DataFrame(data)

# --- Tests ---

@patch("Category_realations.stat_visualization.plt") 
@patch("Category_realations.stat_visualization.sns")
def test_plot_spearman_bar_chart_success(mock_sns, mock_plt, mock_spearman_df):
    """
    Verifies that the Spearman chart plotting function calls the correct plotting methods.
    """
    # Run the function
    plot_spearman_bar_chart(mock_spearman_df)
    
    # 1. Assert that a figure was created
    mock_plt.figure.assert_called_once()
    
    # 2. Assert that seaborn barplot was called with correct data
    mock_sns.barplot.assert_called_once()
    
    # Check arguments passed to barplot
    call_args = mock_sns.barplot.call_args[1] # Get keyword arguments
    assert call_args['x'] == 'Category'
    assert call_args['y'] == 'Spearman Coeff'
    
    # 3. Assert title and labels were set
    mock_plt.title.assert_called()
    mock_plt.xlabel.assert_called_with('Category', fontsize=12)

def test_plot_spearman_empty_df(caplog):
    """
    Verifies that the function handles empty DataFrames gracefully without crashing.
    """
    empty_df = pd.DataFrame(columns=['Category', 'Spearman Coeff'])
    
    # Run function
    plot_spearman_bar_chart(empty_df)
    
    # Check if a warning was logged
    assert "Spearman DataFrame is empty. Skipping plot." in caplog.text

def test_spearman_plot_error_handling(mock_spearman_df, caplog):
    """
    Verifies that exceptions during plotting are caught and logged.
    """
    # We patch plt to raise an exception when 'figure' is called
    with patch("Category_realations.stat_visualization.plt") as mock_plt:
        mock_plt.figure.side_effect = RuntimeError("Plotting crashed!")
        
        # Expect the function to raise the error eventually
        with pytest.raises(RuntimeError):
            plot_spearman_bar_chart(mock_spearman_df)
            
        # Verify the error was logged before re-raising
        assert "An error occurred while plotting" in caplog.text