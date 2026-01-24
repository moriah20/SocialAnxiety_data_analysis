import pytest
import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm

from Category_realations.multiple_linear_regression import run_multiple_regression

# --- Fixtures ---

@pytest.fixture
def regression_data():
    """
    Creates synthetic data for regression testing.
    We use RANDOM data to avoid multicollinearity (correlation between predictors),
    so the model can accurately identify the coefficient for each variable.
    """
    np.random.seed(42) # Set seed so results are always the same
    n_samples = 100
    
    # Independent variables (Predictors) - NOW INDEPENDENT!
    # We use random numbers instead of linspace to ensure they don't depend on each other
    x1 = np.random.rand(n_samples) * 10  # Random values 0-10
    x2 = np.random.rand(n_samples) * 5   # Random values 0-5
    
    # Create the scores DataFrame
    scores_df = pd.DataFrame({
        'Lifestyle': x1,
        'Health': x2
    })

    # Dependent variable (Target) with a strict linear relationship
    # Formula: Anxiety = 2*Lifestyle + 3*Health + 5
    y = 2 * x1 + 3 * x2 + 5
    
    # Create original DataFrame
    original_df = pd.DataFrame({
        'Anxiety Level (1-10)': y,
        'Irrelevant_Column': np.random.rand(n_samples)
    })

    return scores_df, original_df

# --- Tests ---

def test_regression_success_and_coefficients(regression_data):
    """
    Verifies that the regression runs successfully and produces accurate coefficients.
    """
    scores_df, original_df = regression_data
    
    # Run the function
    model = run_multiple_regression(scores_df, original_df, target_col="Anxiety Level (1-10)")

    # 1. Verify the return type (Should be a statsmodels RegressionResultsWrapper)
    assert hasattr(model, 'params'), "The returned object should have a 'params' attribute."
    assert hasattr(model, 'rsquared'), "The returned object should have an 'rsquared' attribute."

    # 2. Verify that a constant (intercept) was added
    assert 'const' in model.params, " The model should include a constant (intercept)."

    # 3. Verify the accuracy of the coefficients (Logic Check)
    # We expect: const ~ 5, Lifestyle ~ 2, Health ~ 3
    # We use a small tolerance since floating point math can be slightly off
    assert model.params['const'] == pytest.approx(5.0, abs=0.01)
    assert model.params['Lifestyle'] == pytest.approx(2.0, abs=0.01)
    assert model.params['Health'] == pytest.approx(3.0, abs=0.01)
    
    # 4. Verify R-squared is 1.0 (since our data is perfect)
    assert model.params['Health'] == pytest.approx(3.0, abs=0.01)
    assert model.rsquared == pytest.approx(1.0, abs=0.0001)

def test_regression_missing_target_error(regression_data, caplog):
    """
    Verifies that a ValueError is raised and logged when the target column is missing.
    """
    scores_df, original_df = regression_data
    wrong_col = "Non_Existent_Target"

    # Expect a ValueError to be raised
    with pytest.raises(ValueError, match="not found"):
        run_multiple_regression(scores_df, original_df, target_col=wrong_col)

    # Verify error logging
    assert f"Target column '{wrong_col}' not found" in caplog.text
    assert any(record.levelname == 'ERROR' for record in caplog.records)

def test_regression_log_summary_output(regression_data, caplog):
    """
    Verifies that the regression summary table is correctly sent to the logger
    instead of being printed to stdout.
    """
    scores_df, original_df = regression_data
    
    # Set log level to INFO to ensure we capture the output
    caplog.set_level(logging.INFO)
    
    run_multiple_regression(scores_df, original_df)

    # 1. Check for the specific log message preceding the table
    assert "Regression Results:" in caplog.text
    
    # 2. Check for standard Statsmodels summary headers in the log
    # These strings always appear in model.summary() output
    assert "OLS Regression Results" in caplog.text
    assert "Dep. Variable:" in caplog.text
    assert "R-squared:" in caplog.text
    
    # 3. Ensure no errors were logged
    assert "ERROR" not in caplog.text