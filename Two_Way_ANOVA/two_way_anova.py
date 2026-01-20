import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
from statsmodels.formula.api import ols
import pingouin as pg
import logging

logger = logging.getLogger(__name__)

# Performs Two-Way ANOVA and returns the summary table.
def two_way_anova_test(df, target_col, IV1, IV2, col_name):
    """
    Performs a Two-Way ANOVA to test the effects of two categorical IVs on a numeric DV.
    """
    # 1. Validate input types: Ensure df is a DataFrame and column names are strings
    if isinstance(df, pd.DataFrame) and all(isinstance(i, str) for i in [IV1, IV2, target_col, col_name]):
        
        logger.info(f"Starting Two-Way ANOVA: {IV1} and {IV2} predicting {col_name}")

        # 2. Data Cleaning: Strip any leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()

        # 3. Target Variable Processing: Safely convert the target column to numeric values
        # 'coerce' will turn non-numeric values into NaN (Not a Number)
        df[col_name] = pd.to_numeric(df[target_col], errors='coerce')

        # 4. Handle Missing Values: Remove rows with NaN in the target or independent variables
        initial_count = len(df)
        df = df.dropna(subset=[col_name, IV1, IV2])
        after_count = len(df)
        
        logger.info(f"Data cleaning complete. Rows before: {initial_count}, Rows after dropna: {after_count}")

        if after_count == 0:
            logger.error("Error: No data remaining after dropna. Cannot perform ANOVA.")
            return None

        # 5. Statistical Analysis: Perform Two-Way ANOVA including main effects and interaction
        # dv = Dependent Variable, between = Independent Variables (Factors)
        anova_table = pg.anova(dv=col_name, between=[IV1, IV2], data=df)

        # 6. Output Results: Log the ANOVA summary table
        logger.info("\n" + "="*30 + "\nANOVA TABLE\n" + "="*30 + f"\n{anova_table.to_string()}")
        
        return anova_table # Returning the table for further use if needed
        
    else:
        # Error handling if input types are incorrect
        logger.error("Error: Wrong variable types. Please ensure df is a DataFrame and names are strings.")
        return None

# Generates an interaction plot for visualization.
def plot_anova_results(df, IV1, IV2, DV):
    """
    Creates an interaction plot to visualize the relationship between two IVs and the DV.
    """
    try:
        logger.info(f"Generating interaction plot for {IV1} and {IV2}")
        
        plt.figure(figsize=(10, 6))
        
        # Using pointplot to show the interaction between the two factors
        # dodge=True prevents points from overlapping
        sns.pointplot(data=df, x=IV1, y=DV, hue=IV2, dodge=True, markers=['o', 's'], capsize=.1)
        
        # Formatting the plot
        plt.title(f'Interaction Plot: {IV1} & {IV2} on {DV}', fontsize=14)
        plt.xlabel(IV1, fontsize=12)
        plt.ylabel(f'Mean {DV}', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Saving the plot as an image
        filename = f"interaction_plot_{IV1}_{IV2}.png"
        plt.savefig(filename)
        
        logger.info(f"Interaction plot displayed and saved as {filename}")
        plt.show()
        
    except Exception as e:
        logger.error(f"Could not create interaction plot: {e}")

# Performs simple main effects when an interaction is found.
def run_post_hoc_analysis(df, DV, IV1, IV2):
    """
    Performs pairwise comparisons (Post-hoc) to identify specific group differences.
    """
    logger.info(f"Running Post-hoc analysis: Simple main effects for {IV1} within each {IV2}")

    try:
        # Simple main effect of IV1 within each level of IV2 using Bonferroni correction
        simple_effects = pg.pairwise_tests(data=df, dv=DV, 
                                          between=[IV1, IV2], 
                                          padjust='bonferroni')
        
        logger.info(f"--- Simple Main Effects Results ({IV1} within {IV2}) ---")
        logger.info(f"\n{simple_effects.to_string()}")
        
        return simple_effects
    
    except Exception as e:
        logger.error(f"An error occurred during Post-hoc analysis: {e}")
        return None
