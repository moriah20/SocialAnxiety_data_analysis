import statsmodels.api as sm
import pandas as pd

def run_multiple_regression(scores_df, original_df, target_col="Anxiety Level (1-10)"):
    """
    Performs Multiple Linear Regression to predict Anxiety Level based on category scores.
    
    Args:
        scores_df: DataFrame containing the independent variables (Lifestyle, Health, History scores).
        original_df: DataFrame containing the dependent variable (Anxiety).
        target_col: Name of the target column.
        
    Returns:
        model_summary: The statistical summary of the regression model.
    """
    
    # 1. Define the Dependent Variable (Y) - The target we want to predict
    y = original_df[target_col]
    
    # 2. Define the Independent Variables (X) - The scores we calculated
    X = scores_df.copy()
    
    # 3. Add a constant (Intercept) to the model
    # (Statsmodels requires this manually, unlike other libraries)
    X = sm.add_constant(X)
    
    # 4. Fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(y, X).fit()
    
    # 5. Print the comprehensive summary table
    print(model.summary())
    
    return model

# --- How to run this ---
# Assuming you have 'scores_df' (X) and 'data' (Y) from previous steps:
