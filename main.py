import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
from statsmodels.formula.api import ols
import pingouin as pg
from scipy.stats import spearmanr 
from binarization.binarize_category import apply_binary
from binarization.health_ranges import create_health_config
from Categorization.Categorization import categorization
from Statistic_Analysis_for_Binaraziation.spearman_test import spearman_test
from Statistic_Analysis_for_Binaraziation.multiple_linear_regression import run_multiple_regression
from Regression.regression import calculate_age_regression
from Regression.regression import plot_regression
from Regression.regression import extract_correlation_from_regression
from Regression.regression import interpret_regression_significance
from Two_Way_ANOVA import two_way_anova_test
from Two_Way_ANOVA import run_post_hoc_analysis
from Two_Way_ANOVA.interctions import main_effects
from Two_Way_ANOVA.interctions import check_effects
from Two_Way_ANOVA.interctions import main_effects_plots
from Two_Way_ANOVA.interctions import calculate_interaction
from Two_Way_ANOVA.interctions import plot_interaction

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

main_effect_1,main_effect_2=main_effects(df, "Gender", "Occupation","Anxiety Level (1-10)")
check_effects=check_effects(main_effect_1, main_effect_2)
main_effects_plots(main_effect_1, main_effect_2, "Gender", "Occupation","Anxiety Level (1-10)")
status=calculate_interaction(df, "Anxiety Level (1-10)", "Gender", "Occupation")
plot_interaction(df, "Anxiety Level (1-10)", "Gender", "Occupation")
anova_tabel=two_way_anova_test(df,  "Anxiety Level (1-10)", "Gender", "Occupation", "Anxiety_Score")
run_post_hoc_analysis(df,"Anxiety Level (1-10)", "Gender", "Occupation")
regression_results=calculate_age_regression(df, "Anxiety Level (1-10)", "Age")
plot_regression(df, "Age",  "Anxiety Level (1-10)")
correlation=extract_correlation_from_regression(regression_results)
p_val=interpret_regression_significance(regression_results)
