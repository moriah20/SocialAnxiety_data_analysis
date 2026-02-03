import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
from statsmodels.formula.api import ols
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import logging
from Visualization.visualization_saving_decorator import auto_save_plot

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

@auto_save_plot(output_dir="Visualization") #save and show plot
def main_effects_plots(main_effect_1, main_effect_2, IV1, IV2, DV):
    """
    Generates side-by-side vertical bar plots for main effects.
    Includes logging for process tracking and error handling.
    """
    
    # Validate inputs before proceeding
    if check_effects(main_effect_1, main_effect_2) and all(isinstance(i, str) for i in [IV1, IV2, DV]):
        
        logger.info(f"Initiating main effects plot generation for {IV1} and {IV2}.")

        try:
            # 1. Create the figure and subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f"Main Effects of {IV1} and {IV2} on {DV}", fontsize=18)
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

        except Exception as e:
            logger.error(f"An error occurred during plot generation: {e}", exc_info=True)
    else:
        logger.warning("Data validation failed: Check if main_effect inputs are Series/DataFrames and names are strings.")


def calculate_interaction(df, col_name, IV1, IV2):
    """Performs Two-Way ANOVA and prepares data for visualization."""
    # 1. Basic Validations and Data Cleaning
    if not isinstance(df, pd.DataFrame) or not all(isinstance(i, str) for i in [IV1, IV2, col_name]):
        logger.error("Invalid input types.")
        return None, None

    req_cols = [col_name, IV1, IV2]
    if not all(c in df.columns for c in req_cols):
        logger.error(f"Missing columns: {req_cols}")
        return None, None

    clean_df = df.dropna(subset=req_cols).copy()
    if len(clean_df) < 5:
        logger.error(f"Insufficient data: {len(clean_df)} rows.")
        return None, None

    try:
        logger.info(f"Running Two-Way ANOVA for {IV1} and {IV2} on {col_name}")
        results = pg.anova(data=clean_df, dv=col_name, between=[IV1, IV2])

        # 2. Results Interpretation
        logger.info("\n--- ANOVA Analysis Results ---")
        for _, row in results.iterrows():
            src, p = row['Source'], row['p-unc']
            if src not in [IV1, IV2] and not any(s in src for s in ['*', ':']):
                continue  # Skip residuals

            # Determine label and status
            if src == IV1: label = f"Main Effect of {IV1}"
            elif src == IV2: label = f"Main Effect of {IV2}"
            else: label = f"Interaction Effect ({IV1} x {IV2})"
            
            status = "Significant" if p < 0.05 else "Not Significant"
            logger.info(f"* {label}: {status} (p = {p:.4f})")
        
        return results, clean_df

    except Exception as e:
        logger.error(f"Statistical calculation failed: {str(e)}")
        return None, None

@auto_save_plot(output_dir="Visualization") #save and show plot
def plot_interaction_bar(results, clean_df, IV1, IV2, DV):
    """Generates a bar plot based on calculated ANOVA results."""
    if results is None or clean_df is None:
        logger.warning("No valid results to plot.")
        return

    try:
        # 1. Extract interaction significance
        inter_row = results[results["Source"].str.contains(r"\*|:", regex=True)]
        if inter_row.empty: return
            
        p_val = float(inter_row["p-unc"].values[0])
        is_sig = p_val < 0.05

        # 2. Setup visualization
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=clean_df, x=IV1, y=DV, hue=IV2, errorbar=None, palette='flare')

        # 3. Titles and Annotations
        plt.title(f'Interaction: {IV1} × {IV2} on {DV}', fontsize=14)
        plt.xlabel(IV1); plt.ylabel(f'Mean {DV}')

        # 4. Significance Box
        sig_text = "Significant" if is_sig else "Non-Significant"
        color = "darkgreen" if is_sig else "darkred"
        plt.text(0.02, 0.95, f"{sig_text} Interaction\np = {p_val:.4f}", 
                 transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor=color))

        plt.tight_layout()
        logger.info("Plot generated successfully.")
    except Exception as e:
        logger.error(f"Plot failed: {e}")


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









