import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
from statsmodels.formula.api import ols
import pingouin as pg
import logging

logger = logging.getLogger(__name__)

# Calculates group means for main effects analysis.
def main_effects(df, IV1, IV2, col_name):
    # Validate inputs: Check if df is a DataFrame and column names are strings
    if isinstance(df, pd.DataFrame) and all(isinstance(i, str) for i in [IV1, IV2, col_name]):
        
        logger.info(f"Calculating main effects for {IV1} and {IV2} on {col_name}")
        
        # Calculate Main Effect for IV1: Group by IV1 and compute mean
        main_IV1 = df.groupby(IV1)[col_name].mean().sort_values(ascending=False)
        logger.info(f"Main effect for {IV1} calculated.")
        
        # Calculate Main Effect for IV2: Group by IV2 and compute mean
        main_IV2 = df.groupby(IV2)[col_name].mean().sort_values(ascending=False)
        logger.info(f"Main effect for {IV2} calculated.")
        
        return main_IV1, main_IV2
    
    else:
        # Error handling for incorrect input types
        logger.error("Error: Wrong variable types. Please enter valid DataFrame and strings.")
        return None, None

# Validates that the input effects are Pandas Series with numeric data.
def check_effects(main_effect_1, main_effect_2):
    # Iterate through both effects to verify data types
    for effect in [main_effect_1, main_effect_2]:
        # Check if the variable is a Pandas Series
        if not isinstance(effect, pd.Series):
            logger.error("Error: Input must be a Pandas Series")
            return False   
        # Check if the data within the Series is numeric (int or float)
        if not pd.api.types.is_numeric_dtype(effect):
            logger.error("Error: Series values must be numeric")
            return False       
    # Return True if all checks pass
    return True

# Generates bar plots for main effects visualization.       
def main_effects_plots(main_effect_1, main_effect_2, IV1, IV2, DV):
    
    # Validate that effects are Series and labels (IVs/DV) are strings
    if check_effects(main_effect_1, main_effect_2) and all(isinstance(i, str) for i in [IV1, IV2, DV]):
        
        logger.info(f"Generating main effects plots for {IV1} and {IV2}")
        plt.figure(figsize=(12, 5))

        # Subplot 1: Main effect of the first independent variable
        plt.subplot(1, 2, 1)
        sns.barplot(x=main_effect_1.index, y=main_effect_1.values, palette='viridis')
        plt.title(f'Mean {DV} by {IV1}')
        plt.ylabel(f'Average {DV} Score')

        # Subplot 2: Main effect of the second independent variable
        plt.subplot(1, 2, 2)
        sns.barplot(x=main_effect_2.index, y=main_effect_2.values, palette='magma')
        plt.xticks(rotation=45) # Rotate x-axis labels for better readability
        plt.title(f'Mean {DV} by {IV2}')
        plt.ylabel(f'Average {DV} Score')

        # Final layout adjustments
        plt.tight_layout()
        plt.show()
        logger.info("Plots displayed successfully.")
        
    else:
        # Error handling for incorrect data types
        logger.error("Error: Wrong variable types. Please check your inputs.")

# Analyzes ANOVA results to determine the significance of main effects and interactions.
def calculate_interaction(df, col_name, IV1, IV2):

    # Validate inputs: Ensure df is a DataFrame and column names are strings
    if isinstance(df, pd.DataFrame) and all(isinstance(i, str) for i in [IV1, IV2, col_name]):
        
        logger.info(f"Running Two-Way ANOVA for {IV1} and {IV2} on {col_name}")
        # Run the Two-Way ANOVA
        results = pg.anova(data=df, dv=col_name, between=[IV1, IV2])

        logger.info("\n--- ANOVA Analysis Results ---")

        # Iterate through the ANOVA table to interpret each source of variation
        for index, row in results.iterrows():
            source = row['Source']  # Factor name (IV1, IV2, or Interaction)
            p_val = row['p-unc']    # Uncorrected p-value
            
            # Significance threshold (Alpha = 0.05)
            is_significant = p_val < 0.05
            
            # Identify the type of effect based on the Source column
            if source == IV1:
                type_of_effect = f"Main Effect of {IV1}"
            elif source == IV2:
                type_of_effect = f"Main Effect of {IV2}"
            elif ' * ' in source or ':' in source:
                type_of_effect = f"Interaction Effect ({IV1} x {IV2})"
            else:
                continue

            # Determine significance status message
            status = "Significant" if is_significant else "Not Significant"
            logger.info(f"* {type_of_effect}: {status} (p = {p_val:.4f})")
        
        return results
            
    else:
        # Error handling for incorrect input types
        logger.error("Error: Wrong variable types. Please ensure df is a DataFrame and names are strings.")
        return None












