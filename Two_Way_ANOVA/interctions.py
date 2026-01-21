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
    Generates vertical bar plots for main effects.
    X-axis: Independent Variables (IV1, IV2)
    Y-axis: Dependent Variable (DV) mean scores
    """
    # Clear any previous plots from memory
    plt.close('all')

    # Validate data types using your existing check_effects function
    if check_effects(main_effect_1, main_effect_2) and all(isinstance(i, str) for i in [IV1, IV2, DV]):
        
        logger.info(f"Starting generation of main effects plots for {IV1} and {IV2}")

        try:
            # Create a figure with two subplots side-by-side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # --- Plot 1: Main effect of IV1 (e.g., Gender) ---
            # Sort values descending to show highest scores first
            m1 = main_effect_1.sort_values(ascending=False)
            sns.barplot(ax=ax1, x=m1.index, y=m1.values, palette='viridis')
            
            ax1.set_title(f'Mean {DV} by {IV1}', fontsize=14)
            ax1.set_xlabel(IV1)           # Independent Variable on X
            ax1.set_ylabel(f'Mean {DV}')  # Dependent Variable on Y
            ax1.tick_params(axis='x', rotation=45) # Slight rotation for readability

            # --- Plot 2: Main effect of IV2 (e.g., Occupation) ---
            # Sort values descending
            m2 = main_effect_2.sort_values(ascending=False)
            sns.barplot(ax=ax2, x=m2.index, y=m2.values, palette='magma')
            
            ax2.set_title(f'Mean {DV} by {IV2}', fontsize=14)
            ax2.set_xlabel(IV2)           # Independent Variable on X
            ax2.set_ylabel(f'Mean {DV}')  # Dependent Variable on Y
            
            # Rotate labels to 90 degrees to handle many categories (e.g., 13 occupations)
            ax2.tick_params(axis='x', rotation=90) 

            # Adjust layout to prevent labels from being cut off
            plt.tight_layout()
            
            # Log success and show the plot
            logger.info("Main effects plots generated successfully.")
            plt.show()

        except Exception as e:
            logger.error(f"Failed to generate plots: {str(e)}")
            
    else:
        # Error handling for incorrect input types
        logger.error("Invalid input types provided to main_effects_plots function.")
        print("Error: Please check the logs for more details on variable types.")


# Analyzes ANOVA results to determine the significance of main effects and interactions.
def calculate_interaction(df, col_name, IV1, IV2):
    """
    Performs Two-Way ANOVA and interprets the significance of main effects and interactions.
    """
    # 1. Validate basic input types
    if isinstance(df, pd.DataFrame) and all(isinstance(i, str) for i in [IV1, IV2, col_name]):
        
        # 2. Check if columns actually exist in the DataFrame (Fixes KeyError in tests)
        required_columns = [col_name, IV1, IV2]
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing columns for interaction: {required_columns}")
            return None

        # 3. Check for sufficient data after removing NaNs (Fixes AssertionError in tests)
        clean_df = df.dropna(subset=required_columns)
        if len(clean_df) < 5:
            logger.error(f"Insufficient data for ANOVA: {len(clean_df)} rows found, minimum 5 required.")
            return None

        logger.info(f"Running Two-Way ANOVA for {IV1} and {IV2} on {col_name}")
        
        try:
            # 4. Run the Two-Way ANOVA using Pingouin
            results = pg.anova(data=clean_df, dv=col_name, between=[IV1, IV2])

            logger.info("\n--- ANOVA Analysis Results ---")

            # 5. Iterate through the ANOVA table to interpret results
            for index, row in results.iterrows():
                source = row['Source']  # Factor name
                p_val = row['p-unc']    # Uncorrected p-value
                
                # Significance threshold (Alpha = 0.05)
                is_significant = p_val < 0.05
                
                # Identify effect type
                if source == IV1:
                    type_of_effect = f"Main Effect of {IV1}"
                elif source == IV2:
                    type_of_effect = f"Main Effect of {IV2}"
                elif ' * ' in source or ':' in source:
                    type_of_effect = f"Interaction Effect ({IV1} x {IV2})"
                else:
                    continue

                status = "Significant" if is_significant else "Not Significant"
                logger.info(f"* {type_of_effect}: {status} (p = {p_val:.4f})")
            
            return results

        except Exception as e:
            logger.error(f"Statistical calculation failed: {str(e)}")
            return None
            
    else:
        logger.error("Error: Wrong variable types. Please ensure df is a DataFrame and names are strings.")
        return None

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











