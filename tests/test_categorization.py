import pandas as pd
import pytest

from category_realations.Categorization import categorization


class TestCategorization:

    def setup_method(self):
        """Create sample binary data and guideline table."""
        self.binary_df = pd.DataFrame({
            "smoke": [1, 0, 1],
            "exercise": [0, 1, 1],
            "bmi_normal": [1, 1, 0],
            "bmi_high": [0, 0, 1]
        })

        self.guideline = pd.DataFrame({
            "Category": ["Lifestyle", "Lifestyle", "BMI", "BMI"],
            "Variable": ["smoke", "exercise", "bmi_normal", "bmi_high"]
        })

    # -----------------------------
    # Successful categorization
    # -----------------------------
    def test_categorization_basic(self):
        result = categorization(self.binary_df, self.guideline)

        # Expect 2 categories
        assert list(result.columns) == ["Lifestyle", "BMI"]

        # Lifestyle = mean(smoke, exercise)
        expected_lifestyle = [
            (1 + 0) / 2,
            (0 + 1) / 2,
            (1 + 1) / 2
        ]
        assert result["Lifestyle"].tolist() == expected_lifestyle

        # BMI = mean(bmi_normal, bmi_high)
        expected_bmi = [
            (1 + 0) / 2,
            (1 + 0) / 2,
            (0 + 1) / 2
        ]
        assert result["BMI"].tolist() == expected_bmi

    # -----------------------------
    # Missing variable in binary_df
    # -----------------------------
    def test_missing_variable_error(self):
        bad_guideline = self.guideline.copy()
        bad_guideline.loc[0, "Variable"] = "not_exist"

        with pytest.raises(KeyError):
            categorization(self.binary_df, bad_guideline)

    # -----------------------------
    # Wrong input types
    # -----------------------------
    def test_invalid_binary_df_type(self):
        with pytest.raises(TypeError):
            categorization("not a df", self.guideline)

    def test_invalid_guideline_type(self):
        with pytest.raises(TypeError):
            categorization(self.binary_df, "not a df")

    # -----------------------------
    # Missing required columns
    # -----------------------------
    def test_missing_guideline_columns(self):
        bad_guideline = pd.DataFrame({"Wrong": [1], "Cols": [2]})

        with pytest.raises(ValueError):
            categorization(self.binary_df, bad_guideline)
