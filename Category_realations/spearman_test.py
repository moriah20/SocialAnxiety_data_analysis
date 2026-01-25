import pandas as pd
from scipy.stats import spearmanr
import logging

# Configure logger for the current module
logger = logging.getLogger(__name__)

def spearman_test(scores_df, original_df, target_col="Anxiety Level (1-10)"):
    """
    Performs a Spearman rank correlation test between category scores and the target variable (Anxiety).
    Includes logging for tracking the process and capturing errors.

    Args:
        scores_df (pd.DataFrame): DataFrame containing the computed category scores.
        original_df (pd.DataFrame): The original DataFrame containing the target variable.
        target_col (str): The name of the target column (default: "Anxiety Level (1-10)").

    Returns:
        pd.DataFrame: A DataFrame containing the test results (Correlation Coefficient and P-value).
    """
    logger.info("Starting Spearman Rank Correlation Test...")
    
    try:
        # Validate that the target column exists in the original DataFrame
        if target_col not in original_df.columns:
            error_msg = f"Target column '{target_col}' not found in original DataFrame"
            logger.error(error_msg)
            raise ValueError(error_msg)

        target_series = original_df[target_col]
        results = []

        # Log the initiation of the calculation against the specific target
        logger.info(f"Calculating correlation against target: '{target_col}'")
        
        # Iterate over all columns (categories) in the scores DataFrame
        for col in scores_df.columns:
            # Perform Spearman correlation calculation
            corr, p_val = spearmanr(scores_df[col], target_series)
            
            # Append results to the list
            results.append({
                'Category': col, 
                'Spearman Coeff': corr, 
                'P-value': p_val
            })
            
            # Log as debug info instead of printing
            logger.debug(f"Finished calculating for Category: {col} | Coeff: {corr:.4f} | P-value: {p_val:.4e}")

        # Create the results DataFrame
        results_df = pd.DataFrame(results)

        # Log the final table nicely formatted using to_string()
        # This replaces the manual print loop and keeps the log clean
        logger.info(f"Spearman Correlation Results:\n{results_df.to_string(index=False)}")

        logger.info("Spearman Test completed successfully.")
        return results_df

    except Exception as e:
        # Log the full stack trace if an error occurs
        logger.exception("An error occurred during Spearman Test execution.")
        raise e