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
from regression import calculate_age_regression
from regression import plot_regression
from regression import extract_correlation_from_regression
from regeression import interpret_regression_significance
from two_way_anova import two_way_anova_test
from two_way_anova import run_post_hoc_analysis
from interctions import main_effects
from interctions import check_effects
from interctions import main_effects_plots
from interctions import calculate_interaction
from interctions import plot_interaction


# --- החלק המותאם לדאטה שלך ---

# 1. טעינת הקובץ
df = pd.read_csv('enhanced_anxiety_dataset.csv')

# 2. הגדרת החוקים (כאן את יכולה לשנות את המספרים)
# הפורמט: 'שם_עמודה': (מינימום, מקסימום) או ('==', 'ערך_רצוי')

health_guidelines = {
    # --- מדדים פיזיים (טווחים) ---
    'Sleep Hours': (7, 8),                 # שינה תקינה: 7-9 שעות
    'Heart Rate (bpm)': (60, 100),         # דופק תקין במנוחה
    'Breathing Rate (breaths/min)': (10, 20), # קצב נשימה תקין
    'Diet Quality (1-10)': (5, 10),        # איכות תזונה גבוהה (7 ומעלה)
    'Physical Activity (hrs/week)': (2.5, 5), # פעילות גופנית (לפחות שעתיים וחצי)
    
    # --- מדדים שיש להם גבול עליון (אז שמתי 0 כמינימום) ---
    'Stress Level (1-10)': (1, 3),         # רמת לחץ נמוכה נחשבת בריאה
    'Caffeine Intake (mg/day)': (0, 400),  # צריכת קפאין בטוחה (עד 400 מ"ג)
    'Alcohol Consumption (drinks/week)': (0, 2), # צריכת אלכוהול נמוכה
    'Anxiety Level (1-10)': (1, 3),        # רמת חרדה נמוכה
    'Therapy Sessions (per month)': (1,12),
    'Sweating Level (1-5)': (0,2),

    # --- משתנים של כן/לא ---
    'Smoking': ('==', 'No'),               # לא מעשן = 1
    'Dizziness': ('==', 'No'),             # אין סחרחורות = 1
    'Family History of Anxiety': ('==', 'No'), # ללא היסטוריה משפחתית = 1
    'Recent Major Life Event': ('==', 'No')    # ללא אירוע מטלטל לאחרונה = 1
}
dicti_categ = {"Lifestyle Factors":
                ["Physical Activity (hrs/week)", "Sleep Hours", "Alcohol Consumption (drinks/week)","Diet Quality (1-10)","Caffeine Intake (mg/day)","Smoking"],
                "Health & Physiological Indicators": 
                ["Heart Rate (bpm)", "Breathing Rate (breaths/min)", "Stress Level (1-10)", "Sweating Level (1-5)","Dizziness" ],
                "Mental Health History": 
                ["Family History of Anxiety", "Therapy Sessions (per month)", "Recent Major Life Event"]
                }

#binarize and categorize
config = create_health_config(df, health_guidelines)
binarized_df = apply_binary(df, config)
categorized_df = categorization(binarized_df,dicti_categ)


#print(pd.DataFrame(config))
#print(categorized_df)

#stats spearman
spearman_test_results = spearman_test(categorized_df, df, target_col="Anxiety Level (1-10)" )
print(spearman_test_results)

#stats multi regression
multi = run_multiple_regression(categorized_df, df, target_col="Anxiety Level (1-10)")

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
