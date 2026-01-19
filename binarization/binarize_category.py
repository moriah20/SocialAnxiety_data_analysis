import numpy as np
import pandas as pd

def apply_binary(df, guidelines):
    df = df.copy()  # לא לשנות את המקור
    mapping = {"yes": 1, "no": 0}
    try: 
        for i, category in guidelines["Variable"].items():
            try:
               random = df[category] 
            except:
                 raise IndexError('database and guideline not aligned')
        
            if guidelines["Type"][i].lower() == "binary":
                df[category] = df[category].astype(str).str.strip().str.lower()
                df[category] = df[category].replace(mapping)
                if guidelines["Condition"][i].lower() == "no":
                    df[category] = df[category].replace({0:1, 1:0})


                
            elif guidelines["Type"][i].lower() == "numeric":
                   df[category] = df[category].between(guidelines["Min"][i], guidelines["Max"][i]).astype(int)

            
        return df[list(guidelines["Variable"])]

    except:
        raise ImportError('add exception')



data = pd.read_csv("enhanced_anxiety_dataset.csv")
guidelines = pd.read_excel("health_guidelines.xlsx")


