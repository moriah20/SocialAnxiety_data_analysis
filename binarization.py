import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("enhanced_anxiety_dataset.csv")
#print(data.describe())

#print((data.isnull().sum(axis= 0) / len(data)) * 100)

#print(f"the data has {len(data)} rows, and {len(data.columns)} colums")


#print(f"שכיחות של storke {(data["stroke"].sum()) *100/5110}")
#list = [data["gender"], data["smoking_status"], data["ever_married"], data["Residence_type"]]
#for colum in list:
 #sns.histplot(colum, )


'''
for colum in data.columns:
    # Calculate percentiles
    if colum != "id" and pd.api.types.is_numeric_dtype(data[colum]):
      lower_percentile = data[colum].quantile(0.1)
      upper_percentile = data[colum].quantile(0.9)

    # Filter out the outliers
      outliers = data[(data[colum] < lower_percentile) | (data[colum] > upper_percentile)]
      print(f'num of outliers for {colum} {len(outliers)}')
      print(f"percentege of outliers: {(len(outliers)/len(data)) * 100 }%")
      #print(f"outliers for {colum}: {outliers}")
      


def outliers(df, column):

    # חישוב הרבעון הראשון והשלישי
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    IQR = q3 - q1

    # הגדרת הגבולות (נוסחה סטנדרטית)

    lower_bound = q1 - 1.5 * IQR
    upper_bound = q3 + 1.5 * IQR


    # Filter out the outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    print(f"--- outliers for '{column}': ----")
    print(f'number of outliers {len(outliers)}')
    print(f"percentege: {(len(outliers)/len(df)) * 100 }%")
    return outliers

list1 = ['Age', 'Sleep Hours', 'Physical Activity (hrs/week)',  'Caffeine Intake (mg/day)','Alcohol Consumption (drinks/week)', 'Heart Rate (bpm)', 'Breathing Rate (breaths/min)', 'Stress Level (1-10)', 'Sweating Level (1-5)', 'Therapy Sessions (per month)', 'Diet Quality (1-10)', 'Anxiety Level (1-10)']
for column in list1:
    outliers(data, column)


    
listi = ['gender', 'smoking_status', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',  'stroke']
for a in listi:
    print(data.groupby(a)['Anxiety Level (1-10)'].mean(), "\n")
    '''

ax = sns.histplot(data= data['Anxiety Level (1-10)'], bins= 10 ,stat= 'percent', discrete= 1, color= 'royalblue')
#plt.title('Anxiety Level distribution')
mean_val = data['Anxiety Level (1-10)'].mean()
median_val = data['Anxiety Level (1-10)'].median()

# הוספת קווים אנכיים
plt.axvline(mean_val, color='black', linestyle='--', label=f'Mean: {mean_val:.2f}')
plt.axvline(median_val, color='black', linestyle='-', label=f'Median: {median_val:.2f}')

# הוספת מקרא (Legend)
plt.legend()
plt.title('Distribution of Social Anxiety Levels with Stats')
plt.ylabel('percentage (%)')
#plt.xticks(np.arange(1,11))
plt.show()

print(f'Anxiety Level stats:\n{data['Anxiety Level (1-10)'].describe()}')