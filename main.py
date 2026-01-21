import pandas as pd
import numpy as np
import logging
import sys
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
from statsmodels.formula.api import ols
import pingouin as pg
import os
import logging
from scipy.stats import spearmanr 
from Initial_Data_Analysis.initial_analysis import (is_null, 
                                                    get_outliers_report, 
                                                    plot_outliers, 
                                                    frequency)
from Binarization.binarize_category import apply_binary
from Categorization.Categorization import categorization
from Statistic_Analysis_for_Binaraziation.spearman_test import spearman_test
from Statistic_Analysis_for_Binaraziation.multiple_linear_regression import run_multiple_regression
from Statistic_Analysis_for_Binaraziation.stat_visualization import (plot_spearman_bar_chart, 
                                                                     plot_regression_summary)
from Regression.regression import (calculate_age_regression, 
                                   plot_regression, 
                                   extract_correlation_from_regression, 
                                   interpret_regression)
from Two_Way_ANOVA import two_way_anova as twa
from Two_Way_ANOVA import interctions as inter
def main():
    """
    Main execution pipeline for the Social Anxiety Data Analysis project.
    Flow: Load -> Clean/Categorize -> Correlation -> Regression -> ANOVA -> Interaction.
    """
    # Initialize logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # --- Step 1: Data Acquisition ---
    try:
        df = pd.read_csv('enhanced_anxiety_dataset.csv')
        guideline = pd.read_excel('health_guidelines.xlsx')
        logger.info("Data sources loaded successfully.")
    except Exception as e:
        logger.error(f"Critical Error: Required CSV/Excel files not found. {e}")
        return

    # --- Step 2: Preliminary Exploratory Data Analysis (EDA) ---
    logger.info("Starting initial analysis (nulls, outliers, frequencies)...")
    is_null(df)
    plot_outliers(df)
    frequency(df)
    
    # --- Step 3: Data Transformation (Binarization & Categorization) ---
    logger.info("Applying data transformation based on health guidelines...")
    binarized_df = apply_binary(df, guideline)
    categorized_df = categorization(binarized_df, guideline)

    # --- Step 4: Non-Parametric Correlation (Spearman) ---
    logger.info("Running Spearman rank correlation tests...")
    original_target = "Anxiety Level (1-10)"
    spearman_results = spearman_test(categorized_df, df, target_col=original_target)
    plot_spearman_bar_chart(spearman_results)

    # --- Step 5: Predictive Analysis (Multiple Linear Regression) ---
    logger.info("Executing multiple linear regression analysis...")
    multi_reg = run_multiple_regression(categorized_df, df, target_col=original_target)
    plot_regression_summary(multi_reg)

    # --- Step 6: Inferential Statistics (Two-Way ANOVA & Interactions) ---
    # Rename column to prevent syntax errors in statistical formulas
    df = df.rename(columns={original_target: "Anxiety_Level"})
    dv = "Anxiety_Level"
    iv1, iv2 = "Gender", "Occupation"

    logger.info(f"Analyzing main effects and interactions for {iv1} and {iv2}...")
    
    # Calculate main effects and generate the corrected side-by-side plots
    m1_effect, m2_effect = inter.main_effects(df, iv1, iv2, dv)
    inter.check_effects(m1_effect, m2_effect)
    inter.main_effects_plots(m1_effect, m2_effect, iv1, iv2, dv)

    # Interaction analysis
    inter.calculate_interaction(df, dv, iv1, iv2)
    inter.plot_interaction(df, dv, iv1, iv2)
    
    # Full ANOVA table and Post-Hoc comparisons
    twa.two_way_anova_test(df, dv, iv1, iv2, "Anxiety_Score")
    twa.run_post_hoc_analysis(df, dv, iv1, iv2)
    twa.plot_anova_results(df, iv1, iv2, dv)

    # --- Step 7: Age-Based Regression Analysis ---
    logger.info("Running linear regression for Age vs Anxiety Level...")
    reg_results = calculate_age_regression(df, dv, "Age")
    plot_regression(df, "Age", dv)
    
    # Extract statistics using imported regression logic
    correlation_val = extract_correlation_from_regression(reg_results)
    p_val = interpret_regression(reg_results)
    
    logger.info(f"Analysis Pipeline Complete. Predictor p-value: {p_val}")

if __name__ == "__main__":
    main()
