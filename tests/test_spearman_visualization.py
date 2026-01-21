import pytest
import pandas as pd
import logging
from unittest.mock import patch, MagicMock

from Statistic_Analysis_for_Binaraziation.stat_visualization import plot_spearman_bar_chart

# --- Fixtures ---

@pytest.fixture
def valid_spearman_df():
    """
    Creates a valid dummy DataFrame for testing the plot.
    Includes positive and negative values to test the text positioning logic.
    """
    data = {
        'Category': ['Lifestyle', 'Health', 'History'],
        'Spearman Coeff': [0.5, -0.3, 0.1]
    }
    return pd.DataFrame(data)

# --- Tests ---

@patch('matplotlib.pyplot.show')
@patch('seaborn.barplot')
def test_plot_spearman_success(mock_barplot, mock_show, valid_spearman_df, caplog):
    """
    Test the 'Happy Path':
    1. Verifies that seaborn.barplot is called (the graph is created).
    2. Verifies that plt.show() is called (the window attempts to open).
    3. Verifies that the success log message is recorded.
    
    Note: We mock 'show' and 'barplot' to prevent actual windows from popping up.
    """
    # 1. Run the function
    with caplog.at_level(logging.INFO):
        plot_spearman_bar_chart(valid_spearman_df)
    
    # 2. Verify that seaborn tried to plot data
    # We check if it was called at least once
    mock_barplot.assert_called_once()
    
    # 3. Verify that the function tried to display the plot
    mock_show.assert_called_once()
    
    # 4. Verify logging
    assert "Generating Spearman correlation bar chart..." in caplog.text
    assert "Displaying Spearman plot window." in caplog.text


@patch('matplotlib.pyplot.show')
def test_plot_spearman_empty_df(mock_show, caplog):
    """
    Test the 'Edge Case': Empty DataFrame.
    The function should log a warning and return WITHOUT attempting to plot.
    """
    # 1. Create an empty DataFrame
    empty_df = pd.DataFrame()
    
    # 2. Run the function
    with caplog.at_level(logging.WARNING):
        plot_spearman_bar_chart(empty_df)
    
    # 3. Verify the Warning log
    assert "Spearman DataFrame is empty. Skipping plot." in caplog.text
    
    # 4. Verify that plt.show() was NOT called
    mock_show.assert_not_called()


@patch('seaborn.barplot')
def test_plot_spearman_error_handling(mock_barplot, valid_spearman_df, caplog):
    """
    Test Exception Handling:
    We deliberately force seaborn to crash to ensure the function catches the error,
    logs it properly, and re-raises it.
    """
    # 1. Configure the mock to raise an error when called
    mock_barplot.side_effect = Exception("Simulated Plotting Error")
    
    # 2. Run the function and expect it to fail
    with pytest.raises(Exception, match="Simulated Plotting Error"):
        plot_spearman_bar_chart(valid_spearman_df)
        
    # 3. Verify that the exception was logged with traceback
    assert "An error occurred while plotting" in caplog.text
    # Check if the log level is ERROR
    assert any(record.levelname == "ERROR" for record in caplog.records)