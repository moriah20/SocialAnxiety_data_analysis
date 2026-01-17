import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
from statsmodels.formula.api import ols
import pingouin as pg


#Performs Two-Way ANOVA and returns the summary table.
def two_way_anova_test(df, target_col, IV1, IV2, col_name):
    # 1. Validate input types: Ensure df is a DataFrame and column names are strings
    if isinstance(df, pd.DataFrame) and all(isinstance(i, str) for i in [IV1, IV2, target_col, col_name]):
        
        # 2. Data Cleaning: Strip any leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()

        # 3. Target Variable Processing: Safely convert the target column to numeric values
        # 'coerce' will turn non-numeric values into NaN (Not a Number)
        df[col_name] = pd.to_numeric(df[target_col], errors='coerce')

        # 4. Handle Missing Values: Remove rows with NaN in the target or independent variables
        df = df.dropna(subset=[col_name, IV1, IV2])
        print(f"Number of lines after dropna: {len(df)}")

        # 5. Statistical Analysis: Perform Two-Way ANOVA including main effects and interaction
        # dv = Dependent Variable, between = Independent Variables (Factors)
        anova_table = pg.anova(dv=col_name, between=[IV1, IV2], data=df)

        # 6. Output Results: Print the ANOVA summary table
        print("\n" + "="*30)
        print("ANOVA TABLE")
        print("="*30)
        print(anova_table)
        
        return anova_table # Returning the table for further use if needed
        
    else:
        # Error handling if input types are incorrect
        print("Error: Wrong variable types. Please ensure df is a DataFrame and names are strings.")


#Performs simple main effects when an interaction is found.
def run_post_hoc_analysis(df, DV, IV1, IV2):
    
    # Simple main effect of Employment within each Gender
    simple_effects = pg.pairwise_tests(data=df, dv=DV, 
                                      between=[IV1, IV2], 
                                      padjust='bonferroni')
    
    print(f"--- Simple Main Effects ({IV1} within {IV2}) ---")
    print(simple_effects)
    return simple_effects