import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns

def plot_spearman_bar_chart(spearman_df):
    """
    Plots a clean bar chart of Spearman correlation coefficients.
    Style: Pastel colors, values annotated on bars.
    """
    # הגדרת גודל הגרף
    plt.figure(figsize=(10, 6))
    
    # הגדרת סכמת צבעים (כחול, אפור, כתום - דומה לתמונה ששלחת)
    # את יכולה לשנות את הצבעים כאן אם תרצי
    custom_colors = ['#8da0cb', '#d9d9d9', '#fc8d62'] 
    
    # יצירת הגרף
    ax = sns.barplot(
        x='Category', 
        y='Spearman Coeff', 
        data=spearman_df, 
        palette=custom_colors,
        edgecolor='black', # מסגרת דקה לעמודות
        linewidth=0.5
    )

    # כותרות
    plt.title('Spearman Correlation Coefficients', fontsize=14, pad=20)
    plt.ylabel('Correlation Coefficient (r)', fontsize=12)
    plt.xlabel('Category', fontsize=12)

    # הוספת הערכים המספריים על גבי העמודות
    for p in ax.patches:
        height = p.get_height()
        # בדיקה האם הערך שלילי או חיובי כדי לדעת איפה למקם את הטקסט
        offset = -0.02 if height < 0 else 0.01
        
        ax.text(
            p.get_x() + p.get_width() / 2.,  # מיקום X (אמצע העמודה)
            height + offset,                 # מיקום Y (קצת מתחת/מעל העמודה)
            f'{height:.2f}',                 # הטקסט (המספר עם 2 ספרות אחרי הנקודה)
            ha="center", 
            va="top" if height < 0 else "bottom",
            fontsize=10,
            color='black'
        )

    # קו האפס וסידור הצירים
    plt.axhline(0, color='black', linewidth=0.8) # קו מפריד ב-0
    plt.xticks(rotation=45) # סיבוב הכיתוב למטה שיהיה קריא
    plt.grid(axis='y', linestyle='--', alpha=0.3) # רשת עדינה ברקע
    plt.tight_layout()
    
    # הצגת הגרף
    plt.show()


def plot_regression_summary(model):
    """
    Visualizes the regression coefficients with their 95% Confidence Intervals.
    
    Args:
        model: The fitted statsmodels OLS result object.
    """
    # 1. Extract coefficients and confidence intervals from the model
    # model.conf_int() returns a DataFrame with columns 0 (lower) and 1 (upper)
    results_df = pd.DataFrame({
        'coef': model.params,
        'lower_ci': model.conf_int()[0],
        'upper_ci': model.conf_int()[1]
    })
    
    # 2. Remove the Constant (Intercept) from the plot
    # The intercept is usually a baseline value (e.g., 7.2) and is on a different scale 
    # than the predictors (e.g., -0.5 to -3.0), so plotting them together distorts the graph.
    if 'const' in results_df.index:
        results_df = results_df.drop('const')

    # 3. Calculate error bar sizes
    # errorbar function needs the distance from the point to the top/bottom of the bar
    yerr_lower = results_df['coef'] - results_df['lower_ci']
    yerr_upper = results_df['upper_ci'] - results_df['coef']
    
    # 4. Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot coefficients as points with error bars
    plt.errorbar(
        x=results_df.index, 
        y=results_df['coef'], 
        yerr=[yerr_lower, yerr_upper], 
        fmt='o',            # 'o' = circle marker
        color='#c0392b',    # Dark red color
        ecolor='black',     # Black error bars
        capsize=5,          # Width of the error bar caps
        elinewidth=2,       # Thickness of error bars
        markersize=8
    )
    
    # 5. Add Reference Line at Y=0
    # If an error bar crosses this line, the variable is NOT statistically significant.
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    # 6. Labels and Title
    plt.title('Regression Coefficients: Impact on Anxiety Level', fontsize=14, fontweight='bold')
    plt.ylabel('Coefficient Value (Change in Anxiety)', fontsize=12)
    plt.xlabel('Predictor Categories', fontsize=12)
    
    # 7. Add text annotations for exact values
    for i, txt in enumerate(results_df['coef']):
        plt.annotate(f"{txt:.2f}", 
                     (results_df.index[i], results_df['coef'][i]), 
                     xytext=(15, 0), 
                     textcoords='offset points',
                     fontsize=10, 
                     fontweight='bold')

    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()