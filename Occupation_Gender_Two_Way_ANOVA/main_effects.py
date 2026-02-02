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

def main_effects(df, IV1, IV2, DV):
    """
    Calculates the main effects for two independent variables."""

    logger.info("Starting main effects calculation")

    # Validate input type
    if not isinstance(df, pd.DataFrame):
        logger.error("Invalid input: df is not a pandas DataFrame")
        return None, None

    try:
        # Calculate mean of DV for each level of the first IV
        logger.debug(f"Calculating main effect for {IV1}")
        m1 = df.groupby(IV1)[DV].mean()

        # Calculate mean of DV for each level of the second IV
        logger.debug(f"Calculating main effect for {IV2}")
        m2 = df.groupby(IV2)[DV].mean()

        logger.info("Main effects calculated successfully")

        return m1, m2

    except Exception as e:
        # Log full error traceback for debugging
        logger.exception("An error occurred during main effects calculation")
        return None, None

def check_effects(m1, m2):
    """
    Validates the main effects results before further analysis or plotting.

    This function checks whether both inputs are valid pandas Series
    and ensures they are not empty.
    """

    logger.info("Checking validity of main effects")

    # Check if the first input is a pandas Series
    if not isinstance(m1, pd.Series):
        logger.error("Invalid input: m1 is not a pandas Series")
        return False

    # Check if the second input is a pandas Series
    if not isinstance(m2, pd.Series):
        logger.error("Invalid input: m2 is not a pandas Series")
        return False

    # Check if any of the Series is empty
    if m1.empty or m2.empty:
        logger.warning("One or both main effect Series are empty")
        return False

    logger.info("Main effects validation passed")
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