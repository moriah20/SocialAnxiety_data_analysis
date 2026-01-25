import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from Initial_Data_Analysis.initial_analysis import (
    is_null,
    get_outliers_report,
    plot_outliers,
    frequency
)


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def df_numeric():
    return pd.DataFrame({
        "A": [1, 2, 3, 100],   # Outlier
        "B": [10, 11, 12, 13]
    })


@pytest.fixture
def df_with_nulls():
    return pd.DataFrame({
        "A": [1, None, 3],
        "B": [4, 5, 6]
    })


@pytest.fixture
def df_categorical():
    return pd.DataFrame({
        "Color": ["Red", "Blue", "Red", "Green"],
        "Type": ["A", "B", "A", "C"]
    })


# -----------------------------
# Tests for is_null
# -----------------------------
def test_is_null_true(df_with_nulls):
    assert is_null(df_with_nulls) is True


def test_is_null_false(df_numeric):
    assert is_null(df_numeric) is False


def test_is_null_invalid_input():
    with pytest.raises(Exception):
        is_null(None)


# -----------------------------
# Tests for get_outliers_report
# -----------------------------
def test_get_outliers_report_returns_correct_types(df_numeric):
    outliers, long_df, should_remove = get_outliers_report(df_numeric)

    assert isinstance(outliers, pd.DataFrame)
    assert isinstance(long_df, pd.DataFrame)
    assert isinstance(should_remove, bool)


def test_get_outliers_report_detects_outliers(df_numeric):
    outliers, _, _ = get_outliers_report(df_numeric)
    assert len(outliers) > 0
    assert "reason" in outliers.columns


def test_get_outliers_report_invalid_input():
    with pytest.raises(TypeError):
        get_outliers_report(None)


def test_get_outliers_report_no_numeric_columns():
    df = pd.DataFrame({"A": ["x", "y", "z"]})
    with pytest.raises(ValueError):
        get_outliers_report(df)

# -----------------------------
# Tests for plot_outliers
# -----------------------------
@patch("matplotlib.pyplot.show")  # Prevent actual plotting
def test_plot_outliers_returns_df(mock_show, df_numeric):
    cleaned_df = plot_outliers(df_numeric)
    assert isinstance(cleaned_df, pd.DataFrame)


@patch("matplotlib.pyplot.show")
def test_plot_outliers_invalid_input(mock_show):
    with pytest.raises(TypeError):
        plot_outliers("not a df")


# -----------------------------
# Tests for frequency
# -----------------------------
@patch("matplotlib.pyplot.show")
def test_frequency_valid(mock_show, df_categorical):
    # Should not raise
    frequency(df_categorical)


@patch("matplotlib.pyplot.show")
def test_frequency_no_categorical(mock_show):
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError):
        frequency(df)
