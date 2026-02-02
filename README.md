# Social Anxiety Data Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Data Source](https://img.shields.io/badge/Data-Kaggle-orange)](https://www.kaggle.com/datasets/natezhang123/social-anxiety-dataset)

## Project Description
This project investigates the factors influencing social anxiety levels using a synthetic dataset of 11,000 observations. The analysis explores how individual lifestyle choices, physiological indicators, and demographic variables—specifically age, gender, and occupation—associate with anxiety severity.


### Main Objectives
1.  **Identify Key Drivers:** Determine which categories (Lifestyle, Physiological Health, or Mental Health History) show the strongest statistical association with anxiety.
2.  **Evaluate Age Impact:** Analyze whether age has a meaningful linear association with anxiety levels.
3.  **Interaction Analysis:** Assess the individual and interactive effects of **Gender** and **Occupation** on social anxiety levels.
4.  **Statistical Modeling:** Build models to quantify the impact of each factor category.

### Assumptions & Hypotheses
* **Health & Lifestyle:** We hypothesize that daily habits and physiological indicators influence anxiety levels the most.
* **Age:** We expect a mild negative correlation (anxiety potentially decreasing with age).
* **Gender & Occupation:** We explore if these factors influence anxiety independently or through an interactive effect.

---

##  Project Structure
```text
project/
│
├── data/
│   ├── raw_data.csv            # Original dataset
│   ├── cleaned_data.csv        # Processed data
│   └── health_guidelines.xlsx  # Reference thresholds (NIH/WHO)
│
├── src/
│   ├── data_import.py          # Loading datasets
│   ├── data_cleaning.py        # Handling missing values and formatting
│   ├── outlier_detection.py    # IQR analysis
│   ├── feature_engineering.py  # Binarization based on health standards
│   ├── analysis_correlations.py # Spearman & Pearson correlations
│   ├── regression_models.py    # Multiple Linear Regression
│   ├── anova_analysis.py       # Two-way ANOVA implementation
│   └── visualization.py        # Chart generation
│
├── main.py                     # Entry point for the full pipeline
├── README.md
└── requirements.txt            # List of dependencies
```

---

## Methodology & Key Stages

### 1. Data Processing & Cleaning
* **Integrity Check:** The dataset (11,000 rows, 19 variables) was verified for completeness with **no missing values**.
* **Outlier Detection:** Using the **IQR (Interquartile Range)** method, 666 potential outliers were identified. These were retained do to low impact.


### 2. Statistical Research Questions

### **Q1: Which life category shows the strongest association?**
* **Method:**
* ***Binarization:*** Variables were converted into binary format (0/1) based on **NIH** and **WHO** guidelines. Values within "healthy" ranges were assigned a value of 1.
* ***Classification:*** Variables grouped into three categories: Lifestyle, Mental Health History, Physiological Health.
* ***Statistics:*** Spearman correlations and Multiple Linear Regression were applied.
* **Result:** The model explained **29.6%** of the variance (**R² = 0.296**). **Lifestyle habits** and **Physiological health** were the strongest predictors.

### **Q2: Does age significantly impact social anxiety?**
* **Method:** Pearson correlation and linear regression.
* **Result:** The correlation was statistically significant but **extremely weak (r ≈ -0.01)**, suggesting age is not a meaningful linear predictor here.

### **Q3: Impact of Gender and Occupation?**
* **Method:** **Two-Way ANOVA** to calculate main and interaction effects.
* **Result:** Analysis showed that **neither gender, occupation, nor their interaction** produced a statistically significant effect on anxiety levels.

---

## Dataset Description
* **Source:** [Kaggle - Social Anxiety Dataset](https://www.kaggle.com/datasets/natezhang123/social-anxiety-dataset)
* **Variables:** Demographics, sleep quality, physical activity, caffeine intake, heart rate, and mental health history.
* **Target:** Anxiety Level (Scale 1–10).

---

## Installation & Running

### **1. Clone the repository**
```bash
git clone [https://github.com/moriah20/SocialAnxiety_data_analysis.git](https://github.com/moriah20/SocialAnxiety_data_analysis.git)
cd SocialAnxiety_data_analysis
```
## **2. Install dependencies**
```Bash

pip install -r requirements.txt
```
## **3. Run the complete pipeline**
```Bash

python main.py
```
## Outputs: log file 

## Visuals: All plots are generated and saved in the 'visualization' folder during execution. 
Using the auto_save_plot decorator.

## **References**
* ***NIH:*** Standards for physiological health indicators.
* ***WHO:*** Guidelines for lifestyle and mental health.
* ***Kaggle:*** Dataset documentation and community insights.

Developed by **Moriah, Orin & Aviya**


