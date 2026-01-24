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

# Assuming logger is already configured in your project
logger = logging.getLogger(__name__)

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



def perform_anova(df, col_name, IV1, IV2):
    """
    Validates input data and performs a Two-Way ANOVA using Pingouin.
    
    Args:
        df (pd.DataFrame): The dataset containing the variables.
        col_name (str): The dependent variable (e.g., Anxiety Level).
        IV1 (str): The first independent variable (e.g., Gender).
        IV2 (str): The second independent variable (e.g., Occupation).
        
    Returns:
        pd.DataFrame: The ANOVA table if successful, None otherwise.
    """
    # 1. Validate that input types are correct
    if not isinstance(df, pd.DataFrame) or not all(isinstance(i, str) for i in [IV1, IV2, col_name]):
        logger.error("Error: Invalid input types. df must be a DataFrame and names must be strings.")
        return None

    # 2. Ensure all required columns exist in the DataFrame
    required_columns = [col_name, IV1, IV2]
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Missing columns in DataFrame: {required_columns}")
        return None

    # 3. Remove missing values (NaNs) and check for sufficient sample size
    clean_df = df.dropna(subset=required_columns)
    if len(clean_df) < 5:
        logger.error(f"Insufficient data: Only {len(clean_df)} valid rows found.")
        return None

    try:
        # 4. Execute the Two-Way ANOVA
        logger.info(f"Computing Two-Way ANOVA for {IV1} and {IV2} on {col_name}")
        return pg.anova(data=clean_df, dv=col_name, between=[IV1, IV2])
    except Exception as e:
        logger.error(f"ANOVA calculation failed: {str(e)}")
        return None
    

def interpret_anova_results(results, IV1, IV2):
    """
    Parses the ANOVA table and logs the significance of main effects and interactions.
    
    Args:
        results (pd.DataFrame): The output table from perform_anova.
        IV1 (str): Name of the first independent variable.
        IV2 (str): Name of the second independent variable.
    """
    if results is None:
        logger.warning("No results to interpret.")
        return

    logger.info("\n--- Statistical Interpretation ---")
    
    # Iterate through each row of the ANOVA table
    for index, row in results.iterrows():
        source = row['Source']
        p_val = row['p-unc']
        
        # Determine if the result is statistically significant (Alpha = 0.05)
        is_significant = p_val < 0.05
        status = "Significant" if is_significant else "Not Significant"

        # Categorize the source into Main Effects or Interaction
        if source == IV1:
            effect_label = f"Main Effect of {IV1}"
        elif source == IV2:
            effect_label = f"Main Effect of {IV2}"
        elif any(symbol in source for symbol in ['*', ':']):
            effect_label = f"Interaction Effect ({IV1} x {IV2})"
        else:
            # Skip rows like 'Residual' which don't represent tested effects
            continue

        # Log the final interpretation
        logger.info(f"* {effect_label}: {status} (p-value = {p_val:.4f})")

def plot_interaction(df, IV1, IV2, DV):
    """
    Generates an interaction plot and logs the process.
    Saves the output as a PNG file.
    """
    try:
        logger.info(f"Generating interaction plot for: {IV1} and {IV2} on {DV}")
        
        # 1. Set the visual style
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        
        # 2. Create the interaction plot (point plot)
        # dodge=True prevents markers from overlapping
        plot = sns.pointplot(data=df, x=IV1, y=DV, hue=IV2, 
                             dodge=True, markers=['o', 's'], capsize=.1)
        
        # 3. Add labels and title
        plt.title(f'Interaction Effect: {IV1} x {IV2} on {DV}', fontsize=14)
        plt.xlabel(f'{IV1}', fontsize=12)
        plt.ylabel(f'Mean {DV}', fontsize=12)
        
        # 4. Save the plot to a file
        filename = f"interaction_plot_{IV1}_{IV2}.png"
        plt.savefig(filename)
        
        # 5. Log success
        logger.info(f"Interaction plot successfully created and saved as {filename}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Failed to generate interaction plot: {e}")











