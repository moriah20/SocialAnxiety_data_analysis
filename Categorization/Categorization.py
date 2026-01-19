import pandas as pd
#from Binarization.binarize_category import apply_binary
def categorization(binary_df, guideline):
    categorized_df = {}
    
    # שימוש ב-unique כדי לעבור על כל קטגוריה רק פעם אחת
    unique_categories = guideline["Category"].unique()
    
    for cat in unique_categories:
        # שליחת רשימת כל המשתנים השייכים לקטגוריה הזו
        variables_in_cat = guideline[guideline["Category"] == cat]["Variable"].tolist()
        
        # סכימת העמודות הרלוונטיות מתוך ה-df הבינארי וחישוב ממוצע (אחוז הצלחה)
        # axis=1 סוכם שורות, count נותן את מספר המשתנים בקטגוריה
        categorized_df[cat] = binary_df[variables_in_cat].sum(axis=1) / len(variables_in_cat)

    return pd.DataFrame(categorized_df)


