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
    """Performs Two-Way ANOVA and extracts the interaction p-value."""
    if df is None or df.empty:
        logger.error("Dataframe is empty.")
        return None, None, None

    req_cols = [col_name, IV1, IV2]
    clean_df = df.dropna(subset=req_cols).copy()
    
    if len(clean_df) < 5:
        return None, None, None

    try:
        results = pg.anova(data=clean_df, dv=col_name, between=[IV1, IV2])
        
        # Flexible search: look for a row that contains both IV names
        mask = results['Source'].str.contains(IV1) & results['Source'].str.contains(IV2)
        interaction_row = results[mask]
        
        p_interaction = None
        if not interaction_row.empty:
            p_interaction = interaction_row['p-unc'].values[0]
            
        return results, clean_df, p_interaction

    except Exception as e:
        logger.error(f"ANOVA failed: {e}")
        return None, None, None
    

@auto_save_plot(output_dir="Visualization") #save and show plot
def plot_interaction_bar(df, IV1, IV2, DV, p_val):
    """Generates a bar plot even if p_val is non-significant or None."""
    if df is None or df.empty:
        logger.warning("No data available to plot.")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Create the plot
    sns.barplot(data=df, x=IV1, y=DV, hue=IV2, errorbar=None, palette='flare')

    plt.title(f'Interaction Plot: {IV1} × {IV2}', fontsize=14)
    plt.xlabel(IV1)
    plt.ylabel(f'Mean {DV}')

    # Handle the significance text logic
    if p_val is not None:
        sig_text = "Significant" if p_val < 0.05 else "Non-Significant"
        box_color = "darkgreen" if p_val < 0.05 else "darkred"
        label_text = f"{sig_text} Interaction\np = {p_val:.4f}"
    else:
        box_color = "gray"
        label_text = "Interaction: N/A"

    plt.text(0.02, 0.95, label_text, 
             transform=plt.gca().transAxes, fontsize=11, 
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor=box_color))

    plt.tight_layout()
    plt.show()


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








