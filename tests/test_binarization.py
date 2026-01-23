import pandas as pd
import pytest
# Use a non-GUI backend so matplotlib won't try to open windows during tests
import matplotlib
matplotlib.use("Agg")
from Category_realations.binarize_category import apply_binary  # או מ-main אם לא פיצלת

# Sample guideline dataframe
guidelines = pd.DataFrame({
    "Variable": ["smoking", "exercise", "bmi"],
    "Type": ["binary", "binary", "numeric"],
    "Condition": ["no", "yes", "yes"],
    "Min": [None, None, 18.5],
    "Max": [None, None, 24.9]
})

# Sample input dataframe
sample_df = pd.DataFrame({
    "smoking": ["yes", "no", "no"],
    "exercise": ["yes", "no", "yes"],
    "bmi": [22.0, 30.5, 19.0]
})


def test_apply_binary_basic():
    result = apply_binary(sample_df, guidelines)

    # Check binary transformation with condition "no" (reverse)
    assert result["smoking"].tolist() == [0, 1, 1]

    # Check binary transformation with condition "yes"
    assert result["exercise"].tolist() == [1, 0, 1]

    # Check numeric range binarization
    assert result["bmi"].tolist() == [1, 0, 1]


def test_missing_column_error():
    bad_guidelines = guidelines.copy()
    bad_guidelines.loc[0, "Variable"] = "nonexistent_column"

    with pytest.raises(KeyError):
        apply_binary(sample_df, bad_guidelines)


def test_unknown_type_error():
    bad_guidelines = guidelines.copy()
    bad_guidelines.loc[0, "Type"] = "unsupported_type"

    with pytest.raises(ValueError):
        apply_binary(sample_df, bad_guidelines)
