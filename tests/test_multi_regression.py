import pytest
import pandas as pd
import numpy as np
import logging

from Statistic_Analysis_for_Binaraziation.multiple_linear_regression import run_multiple_regression

# --- Fixtures ---

@pytest.fixture
def regression_data():
    """
    Fixture that provides dummy data for regression testing.
    Returns a tuple: (scores_df, original_df)
    """
    # Create independent variables (X) - 3 categories
    data_x = {
        'Lifestyle Score': [10, 20, 30, 40, 50],
        'Health Score': [5, 15, 25, 35, 45],
        'History Score': [1, 2, 3, 2, 1]
    }
    scores_df = pd.DataFrame(data_x)

    # Create dependent variable (Y) - Target column
    # We create a target that is roughly correlated to X to ensure the model fits
    data_y = {
        'Anxiety Level (1-10)': [2.1, 3.5, 5.0, 7.2, 8.8],
        'Other_Column': ['A', 'B', 'C', 'D', 'E']
    }
    original_df = pd.DataFrame(data_y)

    return scores_df, original_df

# --- Tests ---

def test_run_multiple_regression_success(regression_data):
    """
    Test the successful execution of the regression model.
    Verifies that the model is fitted and contains the expected parameters.
    """
    scores_df, original_df = regression_data
    target_col = 'Anxiety Level (1-10)'

    # Execute the function
    model = run_multiple_regression(scores_df, original_df, target_col=target_col)

    # 1. Verify the returned object is a statsmodels Wrapper (RegressionResults)
    assert model is not None, "The function returned None instead of a model."
    assert hasattr(model, 'params'), "The returned object should have 'params' attribute."
    assert hasattr(model, 'rsquared'), "The returned object should have 'rsquared' attribute."

    # 2. Verify that 'const' (intercept) was added
    # statsmodels adds 'const' when sm.add_constant() is called
    assert 'const' in model.params.index, "The model is missing the Intercept (const)."

    # 3. Verify the number of parameters
    # Should be: number of score columns + 1 (for the constant)
    expected_params_count = len(scores_df.columns) + 1
    assert len(model.params) == expected_params_count, "Incorrect number of model parameters."

def test_run_multiple_regression_missing_column(regression_data):
    """
    Test error handling: Ensure ValueError is raised when the target column is missing.
    """
    scores_df, original_df = regression_data
    
    # Define a target column name that does not exist in the DataFrame
    missing_col = "Non_Existent_Column"

    # Assert that the specific ValueError is raised
    with pytest.raises(ValueError, match=f"Target column '{missing_col}' not found"):
        run_multiple_regression(scores_df, original_df, target_col=missing_col)

def test_regression_logging(regression_data, caplog):
    """
    Test logging: Verify that the function logs info messages and errors correctly.
    'caplog' captures all log records created during the test.
    """
    scores_df, original_df = regression_data
    target_col = 'Anxiety Level (1-10)'

    # --- Scenario A: Check Success Logs ---
    with caplog.at_level(logging.INFO):
        run_multiple_regression(scores_df, original_df, target_col=target_col)
    
    # specific messages we expect to see in the logs
    assert "Starting Multiple Linear Regression analysis..." in caplog.text
    assert "Fitting OLS regression model..." in caplog.text
    assert "Multiple Regression completed successfully." in caplog.text

    # --- Scenario B: Check Error Logs ---
    # Clear previous logs
    caplog.clear() 
    
    missing_col = "Missing_Target"
    
    # Run the function expecting failure
    with pytest.raises(ValueError):
        run_multiple_regression(scores_df, original_df, target_col=missing_col)
    
    # Verify the error was logged with level ERROR
    assert f"Target column '{missing_col}' not found" in caplog.text
    assert any(record.levelname == "ERROR" for record in caplog.records)