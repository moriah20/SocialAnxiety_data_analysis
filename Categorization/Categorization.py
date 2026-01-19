import pandas as pd
from binarization.binarize_category import apply_binary
def categorization(df,category_dict):
    new_columns = {}
    for new_col, old_cols in category_dict.items():
        #vectored column sum and converting to percentage
        new_columns[new_col] = ((df[old_cols].sum(axis = 1))/len(old_cols))
    #casting to df

    return pd.DataFrame(new_columns)


data = pd.read_csv("enhanced_anxiety_dataset.csv")
dicti = {"Lifestyle Factors": ["Physical Activity (hrs/week)", "Alcohol Consumption (drinks/week)","Family History of Anxiety"]}
config = {
    "Physical Activity (hrs/week)": {
        "type": "numeric",
        "healthy_range": [3,5]
        
    },
    "Alcohol Consumption (drinks/week)": {
        "type": "numeric",
        "healthy_range": [5,7]
       
    },
    "Family History of Anxiety": {
        "type": "binary",
        "mapping": {"yes": 1, "no": 0}
    }
}

result = apply_binary(data, config)

#print(f'{categorization(result, dicti).value_counts()}')