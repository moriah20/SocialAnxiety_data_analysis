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
    

def plot_interaction_bar(df, IV1, IV2, DV):
    """
    Creates a bar-plot interaction graph and adds statistical significance
    information based on a Two-Way ANOVA test.
    """

    logger.info(f"Running Two-Way ANOVA for interaction: {IV1} × {IV2} on {DV}")

    # --- Step 1: Clean data ---
    clean_df = df.dropna(subset=[DV, IV1, IV2])

    # --- Step 2: Run Two-Way ANOVA ---
    anova_results = pg.anova(data=clean_df, dv=DV, between=[IV1, IV2])

    # Extract interaction row
    interaction_row = anova_results[anova_results["Source"].str.contains(r"\*|:", regex=True)]
    p_val = float(interaction_row["p-unc"].values[0])
    is_sig = p_val < 0.05

    logger.info(f"Interaction p-value = {p_val:.4f} | Significant = {is_sig}")

    # Text for graph
    sig_text = "Significant Interaction" if is_sig else "Non-Significant Interaction"
    sig_color = "green" if is_sig else "red"

    # --- Step 3: Plot interaction bar chart ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=clean_df,
        x=IV1,
        y=DV,
        hue=IV2,
        ci=None,      # No error bars
        dodge=True
    )

    # Titles and labels
    plt.title(f'Interaction Effect: {IV1} × {IV2} on {DV}', fontsize=14)
    plt.xlabel(IV1, fontsize=12)
    plt.ylabel(f'Mean {DV}', fontsize=12)

    # Legend
    plt.legend(
        title=IV2,
        title_fontsize=12,
        fontsize=10,
        loc="best"
    )

    # --- Step 4: Add significance box on graph ---
    plt.text(
        0.02, 0.95,
        f"{sig_text}\np = {p_val:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        color=sig_color,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor=sig_color)
    )

    plt.tight_layout()
    plt.show()

    return anova_results


def run_post_hoc_tukey(df, DV, IV1, IV2):
    """
    Performs Tukey HSD post-hoc tests for simple effects
    when an interaction between two independent variables is found.
    The function runs Tukey separately within each level of IV2.
    """

    logger.info(f"Running Tukey HSD post-hoc analysis for: {IV1} within each level of {IV2}")

    results = {}

    try:
        # Loop through each level of IV2 (simple effects)
        for level in df[IV2].unique():

            logger.info(f"Processing simple effect for {IV2} = {level}")

            # Subset the data for the current level
            subset = df[df[IV2] == level]

            # Run Tukey HSD
            tukey = pairwise_tukeyhsd(
                endog=subset[DV],     # Dependent variable
                groups=subset[IV1],   # Groups to compare (IV1 levels)
                alpha=0.05
            )

            # Store results
            results[level] = tukey.summary()

            logger.info(f"Tukey HSD completed for {IV2} = {level}")

        return results

    except Exception as e:
        logger.error(f"Error during Tukey post-hoc analysis: {e}")
        return None









