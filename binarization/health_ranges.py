import pandas as pd
import numpy as np

# --- הפונקציות (כמו שהגדרנו קודם) ---

def create_health_config(df, simple_rules):
    """
    יוצרת קונפיגורציה לטווחים ומשתנים בינאריים.
    """
    config = {}
    for col, rule in simple_rules.items():
        if col not in df.columns:
            continue

        # בדיקה אם זה כלל בינארי (מתחיל ב '==')
        if isinstance(rule[0], str) and rule[0] == '==':
            target_value = str(rule[1]).strip().lower()
            unique_values = df[col].astype(str).str.strip().str.lower().unique()
            mapping = {val: (1 if val == target_value else 0) for val in unique_values}
            config[col] = {"type": "binary", "mapping": mapping}
            
        # אחרת: זה טווח מספרי (min, max)
        else:
            config[col] = {
                "type": "range",
                "min": rule[0],
                "max": rule[1]
            }
    return config

def apply_binary(df, config):
    """
    ממירה את הדאטה ל-0 ו-1.
    """
    df = df.copy()
    for category, rules in config.items():
        if rules["type"] == "range":
            # 1 אם בתוך הטווח, 0 אחרת
            df[category] = df[category].between(rules["min"], rules["max"]).astype(int)
            
        elif rules["type"] == "binary":
            col_clean = df[category].astype(str).str.strip().str.lower()
            # 1 אם זה הערך ה"בריא", 0 אחרת
            df[category] = col_clean.map(rules["mapping"]).fillna(0).astype(int)

    return df[list(config.keys())]

# --- החלק המותאם לדאטה שלך ---

# 1. טעינת הקובץ
df = pd.read_csv('enhanced_anxiety_dataset.csv')

# 2. הגדרת החוקים (כאן את יכולה לשנות את המספרים)
# הפורמט: 'שם_עמודה': (מינימום, מקסימום) או ('==', 'ערך_רצוי')

health_guidelines = {
    # --- מדדים פיזיים (טווחים) ---
    'Sleep Hours': (7, 9),                 # שינה תקינה: 7-9 שעות
    'Heart Rate (bpm)': (60, 100),         # דופק תקין במנוחה
    'Breathing Rate (breaths/min)': (12, 20), # קצב נשימה תקין
    'Diet Quality (1-10)': (7, 10),        # איכות תזונה גבוהה (7 ומעלה)
    'Physical Activity (hrs/week)': (2.5, 10), # פעילות גופנית (לפחות שעתיים וחצי)
    
    # --- מדדים שיש להם גבול עליון (אז שמתי 0 כמינימום) ---
    'Stress Level (1-10)': (1, 3),         # רמת לחץ נמוכה נחשבת בריאה
    'Caffeine Intake (mg/day)': (0, 400),  # צריכת קפאין בטוחה (עד 400 מ"ג)
    'Alcohol Consumption (drinks/week)': (0, 4), # צריכת אלכוהול נמוכה
    'Anxiety Level (1-10)': (1, 3),        # רמת חרדה נמוכה

    # --- משתנים של כן/לא ---
    'Smoking': ('==', 'No'),               # לא מעשן = 1
    'Dizziness': ('==', 'No'),             # אין סחרחורות = 1
    'Medication': ('==', 'No'),            # ללא תרופות = 1
    'Family History of Anxiety': ('==', 'No'), # ללא היסטוריה משפחתית = 1
    'Recent Major Life Event': ('==', 'No')    # ללא אירוע מטלטל לאחרונה = 1
}

# 3. יצירת הקונפיגורציה והפעלת הפונקציה
config = create_health_config(df, health_guidelines)
processed_df = apply_binary(df, config)

# 4. הצגת התוצאה
print("הנתונים המקוריים (הצצה):")
print(df[['Sleep Hours', 'Smoking', 'Heart Rate (bpm)']].head())
print("\nהנתונים המעובדים (1=בריא/בטווח, 0=לא בריא):")
print(processed_df.head())

# אם את רוצה לשמור לקובץ חדש:
# processed_df.to_csv('processed_anxiety_data.csv', index=False)