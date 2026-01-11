import numpy as np
import pandas as pd

def apply_binary(df, config):
    df = df.copy()  # לא לשנות את המקור

    for category, rules in config.items():

        if rules["type"] == "binary":
            col_clean = df[category].astype(str).str.strip().str.lower()
            mapping = {k.lower(): v for k, v in rules["mapping"].items()}
            df[category] = col_clean.replace(mapping).astype(int)
            continue

        if rules["type"] == "numeric":
            threshold = rules["threshold"]
            direction = rules["direction"]

            if direction == "higher_is_healthier":
                df[category] = (df[category] >= threshold).astype(int)

            elif direction == "lower_is_healthier":
                df[category] = (df[category] <= threshold).astype(int)

    return df[list(config.keys())]

'''
data = pd.read_csv("enhanced_anxiety_dataset.csv")

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
    "Family History of Anxiety": {
        "type": "binary",
        "mapping": {"yes": 1, "no": 0}
    }
}

result = apply_binary(data, config)
print(result)
'''