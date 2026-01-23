import pytest
import pandas as pd
import numpy as np
import logging

from Category_realations.spearman_test import spearman_test 

# --- Fixtures ---

@pytest.fixture
def input_data():
    """
    Creates dummy data for testing.
    We create a scenario with perfect correlation to easily verify results.
    """
    # Create independent variables (scores)
    data = {
        'Category_A': [1, 2, 3, 4, 5],  # Perfect positive correlation with Target
        'Category_B': [5, 4, 3, 2, 1]   # Perfect negative correlation with Target
    }
    scores_df = pd.DataFrame(data)

    # Create target variable (original data)
    original_data = {
        'Anxiety Level (1-10)': [1, 2, 3, 4, 5],
        'Other_Col': ['x', 'y', 'z', 'w', 'q']
    }
    original_df = pd.DataFrame(original_data)

    return scores_df, original_df

# --- Tests ---

def test_spearman_test_success(input_data):
    """
    Test the 'happy path': Valid inputs should yield a correct DataFrame.
    """
    scores_df, original_df = input_data
    
    # Run the function
    results = spearman_test(scores_df, original_df, target_col="Anxiety Level (1-10)")

    # 1. Verify the result is a DataFrame
    assert isinstance(results, pd.DataFrame), "The function should return a pandas DataFrame."

    # 2. Verify column names exist
    expected_columns = ['Category', 'Spearman Coeff', 'P-value']
    assert list(results.columns) == expected_columns, "Column names do not match requirements."

    # 3. Verify the logic (Correlation values)
    # Category_A should have correlation 1.0 (perfect match)
    row_a = results[results['Category'] == 'Category_A'].iloc[0]
    assert row_a['Spearman Coeff'] == pytest.approx(1.0, 0.01), "Category_A should have perfect positive correlation."

    # Category_B should have correlation -1.0 (perfect inverse match)
    row_b = results[results['Category'] == 'Category_B'].iloc[0]
    assert row_b['Spearman Coeff'] == pytest.approx(-1.0, 0.01), "Category_B should have perfect negative correlation."

def test_spearman_test_missing_column(input_data):
    """
    Test error handling: Should raise ValueError if target column is missing.
    """
    scores_df, original_df = input_data
    
    # Attempt to run with a non-existent column name
    wrong_column = "Non_Existent_Column"
    
    # Verify that it raises ValueError
    with pytest.raises(ValueError, match=f"Target column '{wrong_column}' not found"):
        spearman_test(scores_df, original_df, target_col=wrong_column)

def test_spearman_logging_on_error(input_data, caplog):
    """
    Test logging: Verify that an error message is written to the log when it fails.
    'caplog' is a built-in pytest fixture that captures log messages.
    """
    scores_df, original_df = input_data
    wrong_column = "Missing_Col"

    # We expect a failure here, but we want to check the logs produced before the crash
    with pytest.raises(ValueError):
        spearman_test(scores_df, original_df, target_col=wrong_column)

    # Check if the error was logged
    # We look for the specific error message in the captured logs
    assert f"Target column '{wrong_column}' not found" in caplog.text
    
    # Verify that the log level was actually ERROR
    assert any(record.levelname == "ERROR" for record in caplog.records)