import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
from statsmodels.formula.api import ols
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import logging

logger = logging.getLogger(__name__)
matplotlib.use('Agg') 

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
        main_effect_1 = df.groupby(IV1)[DV].mean()

        # Calculate mean of DV for each level of the second IV
        logger.debug(f"Calculating main effect for {IV2}")
        main_effect_2 = df.groupby(IV2)[DV].mean()

        logger.info("Main effects calculated successfully")

        return main_effect_1, main_effect_2

    except Exception as e:
        # Log full error traceback for debugging
        logger.exception("An error occurred during main effects calculation")
        return None, None

def check_effects(main_effect_1, main_effect_2):
    """
    Validates the statistical main effects results before analysis or plotting.
    
    Ensures that both main effects are provided as valid, non-empty 
    pandas Series objects.
    """

    logger.info("Starting validation for main effect statistical series")

    # 1. Check if inputs were actually provided (not None)
    if main_effect_1 is None or main_effect_2 is None:
        logger.error("Validation failed: One or both effect inputs are None")
        return False

    # 2. Verify that the first effect is a pandas Series
    if not isinstance(main_effect_1, pd.Series):
        logger.error(f"Validation failed: Expected pandas Series, got {type(main_effect_1)}")
        return False

    # 3. Verify that the second effect is a pandas Series
    if not isinstance(main_effect_2, pd.Series):
        logger.error(f"Validation failed: Expected pandas Series, got {type(main_effect_2)}")
        return False

    # 4. Ensure the series contain data
    if main_effect_1.empty or main_effect_2.empty:
        logger.warning("Validation warning: One or both effect series contain no data")
        return False

    logger.info("Main effects successfully validated for plotting/analysis")
    return True

def main_effects_plots(main_effect_1, main_effect_2, IV1, IV2, DV):
    """
    Generates side-by-side vertical bar plots for main effects.
    Includes logging for process tracking and error handling.
    """
    
    # Force non-interactive backend for testing environments
    matplotlib.use('Agg') 


    # Clear any previous figures from memory
    plt.close('all')

    # Validate inputs before proceeding
    if check_effects(main_effect_1, main_effect_2) and all(isinstance(i, str) for i in [IV1, IV2, DV]):
        
        logger.info(f"Initiating main effects plot generation for {IV1} and {IV2}.")

        try:
            # 1. Create the figure and subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            logger.info("Figure and axes created successfully.")

            # --- PLOT 1: IV1 ---
            m1 = main_effect_1.sort_values(ascending=False)
            sns.barplot(x=m1.index, y=m1.values, hue=m1.index, palette='viridis', ax=ax1, legend=False)
            
            ax1.set_title(f'Mean {DV} by {IV1}', fontsize=14)
            ax1.set_xlabel(IV1)
            ax1.set_ylabel(f'Mean {DV}')
            ax1.tick_params(axis='x', rotation=45)
            logger.info(f"First subplot ({IV1}) rendered.")

            # --- PLOT 2: IV2 ---
            m2 = main_effect_2.sort_values(ascending=False)
            sns.barplot(x=m2.index, y=m2.values, hue=m2.index, palette='magma', ax=ax2, legend=False)
            
            ax2.set_title(f'Mean {DV} by {IV2}', fontsize=14)
            ax2.set_xlabel(IV2)
            ax2.set_ylabel(f'Mean {DV}')
            ax2.tick_params(axis='x', rotation=90) 
            logger.info(f"Second subplot ({IV2}) rendered.")

            # 2. Final adjustments
            plt.tight_layout()
            logger.info("Layout adjusted with tight_layout().")
            
            # CRITICAL FOR TESTS: Do not call plt.show() during automated testing
            # plt.show() 
            
            return fig # Returns the figure object for testing purposes

        except Exception as e:
            logger.error(f"An error occurred during plot generation: {e}", exc_info=True)
            return None
    else:
        logger.warning("Data validation failed: Check if inputs are valid.")
        return None