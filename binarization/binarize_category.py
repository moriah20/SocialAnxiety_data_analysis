import numpy as np
import pandas as pd

def apply_binary(df, config):
    df = df.copy()  # לא לשנות את המקור

    for category, rules in config.items():

        if rules["type"] == "binary":
            col_clean = df[category].astype(str).str.strip().str.lower()
            mapping = {k.lower(): v for k, v in rules["mapping"].items()}
            df[category] = col_clean.replace(mapping).astype(int)
            

        elif rules["type"] == "numeric":
            healthy_range = rules.get("healthy_range") #avoids key error if healthy_range ia not defiend.
            if isinstance(healthy_range, (list, tuple)) and len(healthy_range) >= 2:
                df[category] = df[category].between(healthy_range[0], healthy_range[1]).astype(int)

            else:
                raise ValueError(f"Category '{category}' is numeric but healthy_range is not a list: {healthy_range}")

            
    return df[list(config.keys())]


data = pd.read_csv("enhanced_anxiety_dataset.csv")
guidelines = pd.read_excel("health_guidelines.xlsx")


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
print(result)
