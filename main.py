import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
from statsmodels.formula.api import ols
import pingouin as pg
from scipy.stats import spearmanr 
from Binarization.binarize_category import apply_binary
from Binarization.health_ranges import create_health_config
from Categorization.Categorization import categorization
from Statistic_Analysis_for_Binaraziation.spearman_test import spearman_test
from Statistic_Analysis_for_Binaraziation.multiple_linear_regression import run_multiple_regression


# --- החלק המותאם לדאטה שלך ---

# 1. טעינת הקובץ
df = pd.read_csv('enhanced_anxiety_dataset.csv')



#binarize and categorize
#config = create_health_config(df, health_guidelines)
#binarized_df = apply_binary(df, config)
#categorized_df = categorization(binarized_df,dicti_categ)


#print(pd.DataFrame(config))
#print(categorized_df)

#stats spearman
#spearman_test_results = spearman_test(categorized_df, df, target_col="Anxiety Level (1-10)" )
#print(spearman_test_results)

#stats multi regression
#multi = run_multiple_regression(categorized_df, df, target_col="Anxiety Level (1-10)")


