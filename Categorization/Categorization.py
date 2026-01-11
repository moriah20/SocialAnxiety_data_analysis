import pandas as pd
from binarization.binarize_category import apply_binary
def categorization(df,category_dict):
    new_columns = {}
    for new_col, old_cols in category_dict.items():
        #vectored column sum
        new_columns[new_col] = (df[old_cols].sum(axis = 1))
    #casting to return df
    return pd.DataFrame(new_columns)

data = pd.read_csv("enhanced_anxiety_dataset.csv")
dicti = {"Lifestyle Factors": ["Physical Activity (hrs/week)", "Alcohol Consumption (drinks/week)"]}
config = {
    "Physical Activity (hrs/week)": {
        "type": "numeric",
        "threshold": 3,
        "direction": "higher_is_healthier"
    },
    "Alcohol Consumption (drinks/week)": {
        "type": "numeric",
        "threshold": 5,
        "direction": "lower_is_healthier"
    },
    }
result = apply_binary(data, config)

print(categorization(result, dicti))