import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import math
import logging
import os
from scipy import stats

logger = logging.getLogger(__name__)

def calculate_regression(df, target_col, IV):
    """
    Performs a simple linear regression to analyze the impact of Age on the target variable.
    """
    logger.info(f"Starting regression analysis: {IV} predicting {target_col}")
    
    # 1. Validate inputs
    if isinstance(df, pd.DataFrame) and all(isinstance(i, str) for i in [target_col, IV]):
        
        # 2. Data Cleaning
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df[IV] = pd.to_numeric(df[IV], errors='coerce')
        df = df.dropna(subset=[target_col, IV])
        
        # 3. Perform Linear Regression
        regression_results = pg.linear_regression(df[IV], df[target_col])
        
        # Logging results instead of printing
        logger.info("Regression analysis completed successfully.")
        logger.info(f"\n{'='*30}\nREGRESSION RESULTS: {IV} -> {target_col}\n{'='*30}\n{regression_results.to_string()}")
        
        return regression_results
    
    else:
        logger.error("Invalid input types. Please check your DataFrame and column names.")
        return None


def plot_regression(df, predictor_col, outcome_col):
    """
    Creates a scatter plot with a linear regression line and displays the equation.
    """
    # Check if both columns exist in the provided DataFrame
    if predictor_col in df.columns and outcome_col in df.columns:
        logger.info(f"Generating regression plot for {predictor_col} and {outcome_col}")
        
        # Remove missing values to ensure accurate mathematical calculations
        temp_df = df[[predictor_col, outcome_col]].dropna()
        x = temp_df[predictor_col]
        y = temp_df[outcome_col]

        # 1. Calculate the regression line parameters (slope and intercept) using Numpy
        # polyfit(x, y, 1) fits a first-degree polynomial (a straight line)
        slope, intercept = np.polyfit(x, y, 1)
        
        # 2. Calculate the R-squared value to measure goodness of fit
        correlation_matrix = np.corrcoef(x, y)
        r_squared = correlation_matrix[0, 1]**2
        
        # Create the equation string using LaTeX formatting for a professional look
        equation = f'$y = {slope:.2f}x + {intercept:.2f}$\n$R^2 = {r_squared:.2f}$'

        # Initialize the plot
        plt.figure(figsize=(10, 6))
        
        # 3. Create the regression plot using Seaborn
        # The 'ax' object is captured to allow adding text relative to the axes
        ax = sns.regplot(x=x, y=y, 
                         scatter_kws={'alpha':0.5}, 
                         line_kws={'color':'red'})
        
        # 4. Add the equation text box to the plot
        # transform=ax.transAxes ensures coordinates (0.05, 0.95) are relative to the plot area (0 to 1)
        # instead of the raw data values on the axes.
        ax.text(0.05, 0.95, equation, 
                transform=ax.transAxes, 
                fontsize=13, 
                verticalalignment='top', 
                color='black',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Set plot titles and labels
        plt.title(f'Linear Regression: {predictor_col} Predicting {outcome_col}', fontsize=14)
        plt.xlabel(f'{predictor_col}', fontsize=12)
        plt.ylabel(f'{outcome_col}', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.show()
        logger.info("Plot displayed successfully with equation.")
    else:
        logger.error(f"Column '{predictor_col}' or '{outcome_col}' not found.")

def extract_correlation_from_regression(regression_results):
    """
    Extracts r from regression and interprets both direction and strength.
    """
    try:
        r_squared = regression_results.iloc[0]['r2']
        r_magnitude = math.sqrt(r_squared)
        coef = regression_results.iloc[1]['coef']
        
        correlation = r_magnitude if coef > 0 else -r_magnitude
        abs_corr = abs(correlation)
        direction = "positive" if correlation > 0 else "negative"
        
        # Strength logic
        if abs_corr <= 0.3:
            strength = "weak"
        elif abs_corr <= 0.5:
            strength = "moderate"
        else:
            strength = "strong"
            
        message = f"Correlation Analysis: The correlation is a {strength} {direction} correlation ({correlation:.4f})"
        logger.info(message)
        
        return correlation

    except Exception as e:
        logger.error(f"Error extracting correlation: {e}")
        return None
    
    
def interpret_regression(regression_results):
    if regression_results is None:
        return False
    
    # Locate the row for the actual variable (not the Intercept)
    # Usually the second row (index 1)
    if len(regression_results) > 1:
        p_val = regression_results.iloc[1]['pval'] 
        is_significant = p_val < 0.05
        
        logger.info(f"Regression p-value: {p_val:.4f}, Significant: {is_significant}")
        return is_significant
    return False
