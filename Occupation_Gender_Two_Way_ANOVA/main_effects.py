import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
from statsmodels.formula.api import ols
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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

def main_effects_plots(main_effect_1, main_effect_2, IV1, IV2, DV):
    """
    Generates side-by-side vertical bar plots for main effects.
    Includes logging for process tracking and error handling.
    """
    # Clear any previous figures from memory
    plt.close('all')

    # Validate inputs before proceeding
    if check_effects(main_effect_1, main_effect_2) and all(isinstance(i, str) for i in [IV1, IV2, DV]):
        
        logger.info(f"Initiating main effects plot generation for {IV1} and {IV2}.")

        try:
            # 1. Create the figure and subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            logger.info("Figure and axes created successfully.")

            # --- PLOT 1: IV1 (e.g., Gender) ---
            plt.sca(ax1) # Set current axis to the first subplot
            m1 = main_effect_1.sort_values(ascending=False)
            
            sns.barplot(x=m1.index, y=m1.values, hue=m1.index, palette='viridis', ax=ax1, legend=False)
            
            ax1.set_title(f'Mean {DV} by {IV1}', fontsize=14)
            ax1.set_xlabel(IV1)
            ax1.set_ylabel(f'Mean {DV}')
            ax1.tick_params(axis='x', rotation=45)
            logger.info(f"First subplot ({IV1}) rendered.")

            # --- PLOT 2: IV2 (e.g., Occupation) ---
            plt.sca(ax2) # Set current axis to the second subplot
            m2 = main_effect_2.sort_values(ascending=False)
            
            sns.barplot(x=m2.index, y=m2.values, hue=m2.index, palette='magma', ax=ax2, legend=False)
            
            ax2.set_title(f'Mean {DV} by {IV2}', fontsize=14)
            ax2.set_xlabel(IV2)
            ax2.set_ylabel(f'Mean {DV}')
            ax2.tick_params(axis='x', rotation=90) 
            logger.info(f"Second subplot ({IV2}) rendered.")

            # 2. Final adjustments and display
            plt.tight_layout()
            logger.info("Layout adjusted with tight_layout(). Displaying plot.")
            plt.show()

        except Exception as e:
            logger.error(f"An error occurred during plot generation: {e}", exc_info=True)
    else:
        logger.warning("Data validation failed: Check if main_effect inputs are Series/DataFrames and names are strings.")