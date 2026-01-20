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
                                   interpret_regression_significance)
from Two_Way_ANOVA.two_way_anova import (two_way_anova_test, 
                                         run_post_hoc_analysis)
from Two_Way_ANOVA.interctions import (main_effects, 
                                       check_effects, 
                                       main_effects_plots, 
                                       calculate_interaction)



try:
  df = pd.read_csv('enhanced_anxiety_dataset.csv')
  guideline = pd.read_excel('health_guidelines.xlsx')
except:
  raise FileNotFoundError("can't find files")

#initial data analysis
null = is_null(df)
outliers = plot_outliers(df)
variable_frequency = frequency(df)

#binarize and categorize
binarized_df = apply_binary(df, guideline)
categorized_df = categorization(binarized_df,guideline)

#stats spearman
spearman_test_results = spearman_test(categorized_df, df, target_col="Anxiety Level (1-10)" )
plot_spearman_bar_chart(spearman_test_results)

#stats multi regression
multi = run_multiple_regression(categorized_df, df, target_col="Anxiety Level (1-10)")
plot_regression_summary(multi)

'''
main_effect_1,main_effect_2=main_effects(df, "Gender", "Occupation","Anxiety Level (1-10)")
check_effects=check_effects(main_effect_1, main_effect_2)
main_effects_plots(main_effect_1, main_effect_2, "Gender", "Occupation","Anxiety Level (1-10)")
status=calculate_interaction(df, "Anxiety Level (1-10)", "Gender", "Occupation")
#plot_interaction(df, "Anxiety Level (1-10)", "Gender", "Occupation")
anova_tabel=two_way_anova_test(df,  "Anxiety Level (1-10)", "Gender", "Occupation", "Anxiety_Score")
run_post_hoc_analysis(df,"Anxiety Level (1-10)", "Gender", "Occupation")
regression_results=calculate_age_regression(df, "Anxiety Level (1-10)", "Age")
plot_regression(df, "Age",  "Anxiety Level (1-10)")
correlation=extract_correlation_from_regression(regression_results)
p_val=interpret_regression_significance(regression_results)
'''