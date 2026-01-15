import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("enhanced_anxiety_dataset.csv")

def is_null(df):
   null_count = df.isnull().sum(axis= 0)
   if null_count.sum() == 0:
    return False
   else:
      return True


def get_outliers_report(df):
    num_df = df.select_dtypes(include=['number'])
    
    # חישוב גבולות
    q1 = num_df.quantile(0.25)
    q3 = num_df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # הפיכה למבנה ארוך ושמירת אינדקס
    long_df = num_df.reset_index().melt(id_vars='index', var_name='column', value_name='value')

    # הצמדת גבולות וזיהוי חריגים
    long_df['lower'] = long_df['column'].map(lower_bound)
    long_df['upper'] = long_df['column'].map(upper_bound)
    
    outliers_detailed = long_df[
        (long_df['value'] < long_df['lower']) | (long_df['value'] > long_df['upper'])
    ].copy()

    outliers_detailed['reason'] = np.where(
        outliers_detailed['value'] > outliers_detailed['upper'], 'Too High', 'Too Low'
    )
    
    return outliers_detailed, long_df

def plot_outliers(df):
    # 1. הפקת דוח חריגים והנתונים בפורמט ארוך
    outliers_df, long_df_raw = get_outliers_report(df)
    
    # 2. יצירת ה-DataFrame הנקי (בפורמט ארוך לטובת הגרף)
    # אנחנו מוחקים מהטבלה הארוכה את השורות שמופיעות בדוח החריגים
    long_df_clean = long_df_raw.drop(outliers_df.index)
    
    # 3. פונקציית העזר לציור
    def draw(data, title):
        # שימי לב: col="column" ו- x="value" תואמים ל-melt
        g = sns.FacetGrid(data, col="column", col_wrap=4, sharey=False, sharex=False, height=3, aspect=1.2)
        g.map(sns.boxplot, "value", color="skyblue", order=None) # ודאי שזה value קטן
        g.set_titles("{col_name}")
        g.set_xticklabels(rotation=45)
        g.fig.suptitle(title, y=1.05, fontsize=16)
        plt.tight_layout()
        plt.show()

    # 4. הרצת הציורים
    draw(long_df_raw, 'Raw Data (With Outliers)')
    draw(long_df_clean, 'Clean Data (After Removing Outliers)')

    # 5. החזרת ה-DF המקורי נקי (בפורמט רחב)
    idx_to_drop = outliers_df['index'].unique()
    return df.drop(index=idx_to_drop)






def frequency(df):
    # 1. סינון רק של עמודות קטגוריאליות
    cat_df = df.select_dtypes(include=['object', 'category'])
    
    # 2. הפיכת הטבלה למבנה "ארוך" (Long Format) כדי שיהיה קל לצייר
    # זה הופך את כל העמודות לעמודה אחת של "שם המשתנה" ועמודה אחת של "ערך"
    df_melted = cat_df.melt()

    # 3. יצירת גרף עמודות לכל המשתנים בבת אחת
    # col='variable' יוצר גרף נפרד לכל עמודה מקורית בלי לולאת for
    g = sns.catplot(
        data=df_melted, 
        kind="count", 
        x="value", 
        col="variable", 
        col_wrap=2, # כמה גרפים להציג בשורה אחת
        sharex=False, # מאפשר לכל גרף להציג רק את הערכים הרלוונטיים לו
        sharey= False,
        palette="Paired",
        
    )
    g.set_xticklabels(rotation=45)
    g.fig.subplots_adjust(hspace=1.2, bottom=0.2, top=0.9, left=0.1, right=0.9)
    plt.show()
    return None


print(is_null(data))
plot_outliers(data)
frequency(data)

