import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
from statsmodels.formula.api import ols
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from Visualization.visualization_saving_decorator import auto_save_plot
import logging

logger = logging.getLogger(__name__)

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
    
    # Critical: Ensure enough data for 2-way ANOVA
    if len(clean_df) < 5:
        logger.error(f"Insufficient data: {len(clean_df)} rows.")
        return None, None

    try:
        logger.info(f"Running Two-Way ANOVA for {IV1} and {IV2} on {col_name}")
        
        # Performing the actual calculation
        results = pg.anova(data=clean_df, dv=col_name, between=[IV1, IV2])

        # 2. Results Interpretation
        logger.info("\n--- ANOVA Analysis Results ---")
        for _, row in results.iterrows():
            src = row['Source']
            p = row['p-unc']
            
            # Identify the effect type
            if src == IV1:
                label = f"Main Effect of {IV1}"
            elif src == IV2:
                label = f"Main Effect of {IV2}"
            elif "*" in src or ":" in src:
                label = f"Interaction Effect ({IV1} x {IV2})"
            else:
                continue  # Skip Residuals or other sources

            status = "Significant" if p < 0.05 else "Not Significant"
            logger.info(f"* {label}: {status} (p = {p:.4f})")
        
        return results, clean_df

    except Exception as e:
        print(f"\nSTDOUT DEBUG: The ANOVA failed because: {e}") 
        logger.error(f"Analysis failed: {str(e)}")
        #logger.error(f"Statistical calculation failed: {str(e)}")
        # FIX: Consistently return (None, None) on failure
        return None, None
@auto_save_plot(output_dir="Visualization")  # Automatically save and log generated plots
def plot_interaction_bar(df, IV1, IV2, DV, p_val):
    logger.debug(f"Received p_val={p_val} (type={type(p_val)})")

    """
    Generates an interaction bar plot for two categorical independent variables (IV1, IV2)
    and one dependent variable (DV), including a significance annotation based on a 
    pre-calculated p-value.

    Parameters:
        df (pd.DataFrame): The dataset containing the variables.
        IV1 (str): First independent variable (x-axis).
        IV2 (str): Second independent variable (hue).
        DV (str): Dependent variable to be plotted.
        p_val (float): Pre-computed p-value for the interaction effect.

    Notes:
        - The function relies on the auto_save_plot decorator to save the figure.
        - The decorator automatically detects the figure title for filename generation.
    """

    # Validate that data exists
    if df is None or df.empty or p_val is None:
        logger.warning("Plot aborted: Missing data or p-value.")
        return

    # Ensure p_val is numeric
    try:
        p_val = float(p_val)
    except (ValueError, TypeError):
        logger.error(
            f"Plotting failed: p_val must be numeric, received {type(p_val)} with value '{p_val}'"
        )
        return

    try:
        logger.info("Starting interaction plot generation...")

        # Determine statistical significance
        is_sig = p_val < 0.05
        logger.debug(f"Interaction significance evaluated: is_sig={is_sig}, p={p_val}")

        # Set visualization theme
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        logger.debug("Matplotlib figure and axes created.")

        # Create the bar plot
        sns.barplot(
            data=df,
            x=IV1,
            y=DV,
            hue=IV2,
            errorbar=None,
            palette="flare"
        )
        logger.debug("Seaborn barplot rendered successfully.")

        # Use suptitle so the decorator can detect it
        fig.suptitle(f"Interaction: {IV1} × {IV2} on {DV}", fontsize=14)

        # Axis labels
        ax.set_xlabel(IV1)
        ax.set_ylabel(f"Mean {DV}")

        # Add significance annotation box
        sig_text = "Significant" if is_sig else "Non-Significant"
        box_color = "darkgreen" if is_sig else "darkred"

        ax.text(
            0.02, 0.95,
            f"{sig_text} Interaction\np = {p_val:.4f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor=box_color)
        )
        logger.debug("Significance annotation added to plot.")

        # Ensure figure is fully registered before decorator saves it
        fig.canvas.draw()
        logger.info("Interaction plot generated successfully.")

    except Exception as e:
        logger.error(f"Plot generation failed due to unexpected error: {e}", exc_info=True)

def run_post_hoc_tukey(df, DV, IV1, IV2):
    """
    Performs Tukey HSD post-hoc tests for simple effects.
    Runs Tukey separately for each level of IV2 to analyze how IV1 affects the DV.
    """
    
    # 1. Basic Validation - ensures data is provided
    if df is None or df.empty:
        logger.error("Dataframe is empty or None")
        return None

    logger.info(f"Running Tukey HSD for: {IV1} within each level of {IV2}")

    try:
        results = {}  
        
        # 2. Iterate through each level of the second independent variable
        for level in df[IV2].unique():
            # Create a subset for the current level
            subset = df[df[IV2] == level]
            
            # Check if there are at least 2 groups to compare within this level
            if len(subset[IV1].unique()) < 2:
                logger.warning(f"Not enough groups for comparison at level: {level}")
                continue

            # 3. Perform Tukey HSD test
            tukey = pairwise_tukeyhsd(
                endog=subset[DV], 
                groups=subset[IV1], 
                alpha=0.05
            )
            
            # Store the summary table in the results dictionary
            results[level] = tukey.summary()

        # 4. Final check: if no tests were successfully performed, return None
        if not results:
            logger.warning("No valid post-hoc results were generated.")
            return None
            
        return results

    except Exception as e:
        logger.error(f"Error during Tukey post-hoc analysis: {e}")
        return None








