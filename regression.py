
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import math

def calculate_age_regression(df, target_col, IV):
    """
    Performs a simple linear regression to analyze the impact of Age on the target variable.
    """
    # 1. Validate inputs: Ensure df is a DataFrame and columns are strings
    if isinstance(df, pd.DataFrame) and all(isinstance(i, str) for i in [target_col, IV]):
        
        # 2. Data Cleaning: Ensure Age and Target are numeric and drop missing values
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df[IV] = pd.to_numeric(df[IV], errors='coerce')
        df = df.dropna(subset=[target_col, IV])
        
        # 3. Perform Linear Regression
        # X is the predictor (Age), y is the outcome (Target)
        regression_results = pg.linear_regression(df[IV], df[target_col])
        
        print("\n" + "="*30)
        print(f"REGRESSION RESULTS: {IV} predicting {target_col}")
        print("="*30)
        print(regression_results)
        
        return regression_results
    
    else:
        print("Error: Invalid input types. Please check your DataFrame and column names.")
        return None




def plot_regression(df, predictor_col, outcome_col):
    """
    Creates a scatter plot with a linear regression line for any two numeric variables.
    
    Parameters:
    df (pd.DataFrame): The dataset
    predictor_col (str): The Independent Variable (IV) - e.g., 'Age'
    outcome_col (str): The Dependent Variable (DV) - e.g., 'Anxiety_Score'
    """
    # 1. Validate inputs
    if predictor_col in df.columns and outcome_col in df.columns:
        
        plt.figure(figsize=(10, 6))
        
        # 2. Create the plot
        sns.regplot(data=df, x=predictor_col, y=outcome_col, 
                    scatter_kws={'alpha':0.5}, 
                    line_kws={'color':'red'})
        
        # 3. Add dynamic labels and title
        # השתמשתי בשמות המשתנים עצמם כדי שהגרף יתאים לכל דאטה
        plt.title(f'Linear Regression: {predictor_col} Predicting {outcome_col}', fontsize=14)
        plt.xlabel(f'{predictor_col} (Independent Variable)', fontsize=12)
        plt.ylabel(f'{outcome_col} (Dependent Variable)', fontsize=12)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
    else:
        print(f"Error: One of the columns ('{predictor_col}' or '{outcome_col}') not found in DataFrame.")



def extract_correlation_from_regression(regression_results):
    """
    Extracts r from regression and interprets both direction and strength.
    """
    try:
        # 1. Extract statistics
        r_squared = regression_results.iloc[0]['r2']
        r_magnitude = math.sqrt(r_squared)
        coef = regression_results.iloc[1]['coef']
        
        # Determine the actual correlation with its sign
        correlation = r_magnitude if coef > 0 else -r_magnitude
        
        # Calculate absolute value for strength interpretation
        abs_corr = abs(correlation)
        direction = "positive" if correlation > 0 else "negative"
        
        print(f"--- Correlation Analysis ---")
        
        # 2. Strength logic (based on Cohen's criteria)
        if abs_corr <= 0.3:
            strength = "weak"
        elif abs_corr <= 0.5:
            strength = "moderate"
        else:
            strength = "strong"
            
        # 3. Final Output: Combined direction and strength
        # Example: "The correlation is a strong positive correlation: 0.65"
        print(f"The correlation is a {strength} {direction} correlation: {correlation:.4f}")
        
        return correlation

    except Exception as e:
        print(f"Error: {e}")
        return None
    
def interpret_regression_significance(regression_results):
      try:
        # Extract p-value for the predictor (Age)
        # Usually the second row in the table (index 1)
        p_val = regression_results.iloc[1]['p-val']
        
        print(f"--- Significance Analysis ---")
        print(f"P-value: {p_val:.4f}")
        
        if p_val < 0.05:
            print("The relationship is Statistically Significant.")
        else:
            print("The relationship is NOT Statistically Significant (p > 0.05).")
            
        return p_val
      except Exception as e:
        print(f"Could not extract p-value: {e}")
