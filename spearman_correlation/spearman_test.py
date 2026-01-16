import pandas as pd
from scipy.stats import spearmanr

def spearman_test(scores_df, original_df, target_col="Anxiety Level (1-10)"):
    """
    Performs a Spearman rank correlation test between category scores and the target variable (Anxiety).

    Args:
        scores_df: DataFrame containing the summary scores (e.g., Lifestyle Score).
        original_df: The original DataFrame containing the anxiety column.
        target_col: The name of the target column in the original data (default: "Anxiety Level (1-10)").

    Returns:
        DataFrame containing the test results (correlation coefficient and P-value).
    """
    # Check if the target column exists in the original DataFrame
    if target_col not in original_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in original DataFrame")

    target_series = original_df[target_col]
    results = []

    print(f"{'Category':<20} | {'Spearman Coeff':<15} | {'P-value':<15}")
    print("-" * 65)

    # Iterate over all columns in the scores DataFrame
    for col in scores_df.columns:
        # Calculate Spearman correlation
        corr, p_val = spearmanr(scores_df[col], target_series)
        
        # Store and print the results
        results.append({
            'Category': col, 
            'Spearman Coeff': corr, 
            'P-value': p_val
        })
        print(f"{col:<20} | {corr:.4f}{' ':>9} | {p_val:.4e}")

    return pd.DataFrame(results)

# --- Example of how to run this (continuation from your previous code) ---

# Assuming you already have these variables from previous steps:
# 1. data (The original DataFrame)
# 2. scores_df (The DataFrame output from the categorization function)

print("\n--- Running Spearman Test ---")
spearman_results = spearman_test(scores_df, data)