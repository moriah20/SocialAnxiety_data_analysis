# Use a non-GUI backend so matplotlib won't try to open windows during tests
import matplotlib
matplotlib.use("Agg")

import os
import shutil
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
        """Test that is_null returns False when no nulls exist."""
        assert is_null(self.df) == False


    def test_is_null_true(self):
        """Test that is_null returns True when nulls exist."""
        df_with_null = self.df.copy()
        df_with_null.loc[0, "A"] = None
        assert is_null(df_with_null) == True

    # -----------------------------
    # get_outliers_report Tests
    # -----------------------------
    def test_get_outliers_report_detects_outlier(self):
        """Test that outlier detection identifies the outlier in column A."""
        outliers, long_df = get_outliers_report(self.df)

        # Expect at least one outlier (100)
        assert len(outliers) >= 1
        assert "A" in outliers["column"].values

        # Ensure long_df contains expected columns
        assert set(["index", "column", "value", "lower", "upper"]).issubset(long_df.columns)

    # -----------------------------
    # plot_outliers Tests
    # -----------------------------
    @patch("matplotlib.pyplot.show")  # Prevent GUI window
    def test_plot_outliers_runs_without_error(self, mock_show):
        """
        Test that plot_outliers executes without raising errors.
        The function returns None (plots only), so we only check execution.
        """
        result = plot_outliers(self.df)
        assert result is None

    # -----------------------------
    # frequency Tests
    # -----------------------------
    @patch("matplotlib.pyplot.show")  # Prevent GUI window
    def test_frequency_runs_without_error(self, mock_show):
        """Test that frequency plotting runs without raising exceptions."""
        frequency(self.df)
