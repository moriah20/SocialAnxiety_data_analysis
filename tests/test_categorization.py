import pandas as pd
import pytest

from Category_realations.Categorization import categorization


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def binary_df():
    return pd.DataFrame({
        "smoke": [1, 0, 1],
        "exercise": [0, 1, 1],
        "bmi_normal": [1, 1, 0],
        "bmi_high": [0, 0, 1]
    })


@pytest.fixture
def guideline():
    return pd.DataFrame({
        "Category": ["Lifestyle", "Lifestyle", "BMI", "BMI"],
        "Variable": ["smoke", "exercise", "bmi_normal", "bmi_high"]
    })


# -----------------------------
# Successful categorization
# -----------------------------
def test_categorization_basic(binary_df, guideline):
    result = categorization(binary_df, guideline)

    assert list(result.columns) == ["Lifestyle", "BMI"]

    expected_lifestyle = [
        (1 + 0) / 2,
        (0 + 1) / 2,
        (1 + 1) / 2
    ]
    assert result["Lifestyle"].tolist() == expected_lifestyle

    expected_bmi = [
        (1 + 0) / 2,
        (1 + 0) / 2,
        (0 + 1) / 2
    ]
    assert result["BMI"].tolist() == expected_bmi


# -----------------------------
# Missing variable in binary_df
# -----------------------------
def test_missing_variable_error(binary_df, guideline):
    bad_guideline = guideline.copy()
    bad_guideline.loc[0, "Variable"] = "not_exist"

    with pytest.raises(KeyError):
        categorization(binary_df, bad_guideline)


# -----------------------------
# Wrong input types
# -----------------------------
def test_invalid_binary_df_type(guideline):
    with pytest.raises(TypeError):
        categorization("not a df", guideline)


def test_invalid_guideline_type(binary_df):
    with pytest.raises(TypeError):
        categorization(binary_df, "not a df")


# -----------------------------
# Missing required columns
# -----------------------------
def test_missing_guideline_columns(binary_df):
    bad_guideline = pd.DataFrame({"Wrong": [1], "Cols": [2]})

    with pytest.raises(ValueError):
        categorization(binary_df, bad_guideline)
