import statsmodels.api as sm
import pandas as pd
import logging


# Configure logger for the current module
logger = logging.getLogger(__name__)

def run_multiple_regression(scores_df, original_df, target_col="Anxiety Level (1-10)"):
    """
    Performs Multiple Linear Regression to predict Anxiety Level based on category scores.
    Includes logging for tracking process and errors.
    
    Args:
        scores_df: DataFrame containing the independent variables (Lifestyle, Health, History scores).
        original_df: DataFrame containing the dependent variable (Anxiety).
        target_col: Name of the target column.
        
    Returns:
        model: The fitted statsmodels OLS result object.
    """
    logger.info("Starting Multiple Linear Regression analysis...")
    
    try:
        # 1. Validate Data Availability
        # Ensure the target column actually exists before proceeding
        if target_col not in original_df.columns:
            error_msg = f"Target column '{target_col}' not found in original DataFrame"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 2. Define the Dependent Variable (Y) - The target we want to predict
        y = original_df[target_col]
        
        # 3. Define the Independent Variables (X) - The scores we calculated
        X = scores_df.copy()
        
        # 4. Add a constant (Intercept) to the model
        # (Statsmodels requires this manually to calculate the intercept term)
        X = sm.add_constant(X)
        logger.debug("Constant (intercept) added to independent variables.")
        
        # 5. Fit the Ordinary Least Squares (OLS) model
        logger.info("Fitting OLS regression model...")
        model = sm.OLS(y, X).fit()
        
        # 6. Print the comprehensive summary table
        print(model.summary())
        
        logger.info("Multiple Regression completed successfully.")
        return model

    except Exception as e:
        # Log the full stack trace for debugging purposes
        logger.exception("An error occurred during Multiple Regression analysis.")
        raise e