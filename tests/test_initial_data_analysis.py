# Use a non-GUI backend so matplotlib won't try to open windows during tests
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from Initial_Data_Analysis.initial_analysis import (
    is_null,
    get_outliers_report,
    plot_outliers,
    frequency
)


class TestInitialAnalysis:

    def setup_method(self):
        """Create sample data for all tests."""
        self.df = pd.DataFrame({
            "A": [1, 2, 3, 100],   # 100 is an outlier
            "B": [5, 6, 7, 8],
            "C": ["yes", "no", "yes", "no"]
        })

    # -----------------------------
    # is_null Tests
    # -----------------------------
    def test_is_null_false(self):
        # Use == instead of "is" because numpy.bool_ is not identical to Python bool
        assert is_null(self.df) == False

    def test_is_null_true(self):
        df_with_null = self.df.copy()
        df_with_null.loc[0, "A"] = None
        assert is_null(df_with_null) == True

    # -----------------------------
    # get_outliers_report Tests
    # -----------------------------
    def test_get_outliers_report_detects_outlier(self):
        outliers, long_df = get_outliers_report(self.df)
        # Expect at least one outlier (100)
        assert len(outliers) >= 1
        assert "A" in outliers["column"].values

    # -----------------------------
    # plot_outliers Tests
    # -----------------------------
    @patch("matplotlib.pyplot.show")  # Prevents GUI window
    def test_plot_outliers_removes_rows(self, mock_show):
        cleaned_df = plot_outliers(self.df)
        # Original has 4 rows, cleaned should have fewer
        assert len(cleaned_df) < len(self.df)

    # -----------------------------
    # frequency Tests
    # -----------------------------
    @patch("matplotlib.pyplot.show")  # Prevents GUI window
    def test_frequency_runs_without_error(self, mock_show):
        # Should not raise any exception
        frequency(self.df)
