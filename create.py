import pandas as pd

# המילונים המקוריים שלך
health_guidelines = {
    'Sleep Hours': (7, 8),
    'Heart Rate (bpm)': (60, 100),
    'Breathing Rate (breaths/min)': (10, 20),
    'Diet Quality (1-10)': (5, 10),
    'Physical Activity (hrs/week)': (2.5, 5),
    'Stress Level (1-10)': (1, 3),
    'Caffeine Intake (mg/day)': (0, 400),
    'Alcohol Consumption (drinks/week)': (0, 2),
    'Therapy Sessions (per month)': (1, 12),
    'Sweating Level (1-5)': (0, 2),
    'Smoking': ('==', 'No'),
    'Dizziness': ('==', 'No'),
    'Family History of Anxiety': ('==', 'No'),
    'Recent Major Life Event': ('==', 'No')
}

dicti_categ = {
    "Lifestyle Factors": ["Physical Activity (hrs/week)", "Sleep Hours", "Alcohol Consumption (drinks/week)", "Diet Quality (1-10)", "Caffeine Intake (mg/day)", "Smoking"],
    "Health & Physiological Indicators": ["Heart Rate (bpm)", "Breathing Rate (breaths/min)", "Stress Level (1-10)", "Sweating Level (1-5)", "Dizziness"],
    "Mental Health History": ["Family History of Anxiety", "Therapy Sessions (per month)", "Recent Major Life Event"]
}

# 1. יצירת רשימה שטוחה שתחבר בין הקטגוריה למדד ולערכים
data_list = []

# נעבור על הקטגוריות
for category, indicators in dicti_categ.items():
    for indicator in indicators:
        if indicator in health_guidelines:
            guideline = health_guidelines[indicator]
            
            # פירוק הערכים (טווח או תנאי)
            val_min = guideline[0]
            val_max = guideline[1]
            
            data_list.append({
                'Category': category,
                'Indicator': indicator,
                'Min/Value': val_min,
                'Max/Condition': val_max
            })

# 2. יצירת ה-DataFrame
df_guidelines = pd.DataFrame(data_list)

# 3. שמירה לקובץ אקסל
df_guidelines.to_excel('health_guidelines.xlsx', index=False)

print("הקובץ 'health_guidelines.xlsx' נוצר בהצלחה!")
print(df_guidelines.head())