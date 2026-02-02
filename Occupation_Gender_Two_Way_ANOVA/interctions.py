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
        return 0.05, None


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
        plt.show()
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









