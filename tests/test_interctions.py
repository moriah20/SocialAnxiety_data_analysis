import pytest
import pandas as pd
import numpy as np
from statsmodels.stats.multicomp import MultiComparison
from Occupation_Gender_Two_Way_ANOVA.interctions import  calculate_interaction,plot_interaction_bar,run_post_hoc_tukey

@pytest.fixture
def anova_data():
    """Generates a valid dummy dataset for ANOVA testing."""
    return pd.DataFrame({
        'Gender': ['Male', 'Male', 'Female', 'Female'] * 5,
        'Treatment': ['GroupA', 'GroupB', 'GroupA', 'GroupB'] * 5,
        'Score': [80, 70, 85, 75] * 5
    })

# --- Tests for calculate_interaction (The ANOVA wrapper) ---

def test_calculate_interaction_success(anova_data):
    """Verifies that pg.anova is called and returns a results table."""
    results = calculate_interaction(anova_data, 'Score', 'Gender', 'Treatment')
    assert isinstance(results, pd.DataFrame)
    assert 'Source' in results.columns
    assert 'p-unc' in results.columns


def test_calculate_interaction_error_handling():
    """
    Tests if the function handles invalid input types.
    Expected behavior: Returns (None, None) for failed calculation.
    """
    # Passing a string instead of a DataFrame to trigger error handling
    results = calculate_interaction("Not a DataFrame", 'IV1', 'IV2', 'DV')
    
    # Updated: Checking for a tuple of two Nones
    assert results == (None, None)


def test_plot_interaction_runs_without_error(sample_anova_results, sample_df):
    """Checks if the plotting function completes without crashing."""
    try:
        plot_interaction_bar(sample_anova_results, sample_df, 'Gender', 'Occupation', 'Anxiety')
        plt.close('all') # Cleanup
    except Exception as e:
        pytest.fail(f"Plotting function raised an error: {e}")

import pytest
import pandas as pd

# Ensure you import your function correctly
# from your_module import run_post_hoc_tukey
def tukey_data():
    """Generates synthetic data for Tukey HSD testing."""
    return pd.DataFrame({
        'Anxiety_Level': [3.0, 3.2, 5.0, 5.2, 2.0, 2.1, 4.0, 4.3],
        'Gender': ['Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female'],
        'Occupation': ['Doctor', 'Doctor', 'Doctor', 'Doctor', 'Artist', 'Artist', 'Artist', 'Artist']
    })

def test_run_post_hoc_tukey_structure(tukey_data):
    """
    Verifies that the function returns a dictionary where keys are IV2 levels
    and values are Tukey summary tables.
    """
    results = run_post_hoc_tukey(tukey_data, 'Anxiety_Level', 'Gender', 'Occupation')
    
    # 1. Check if the output is a dictionary
    assert isinstance(results, dict)
    
    # 2. Check if all levels of IV2 (Occupation) are present in the keys
    expected_levels = set(tukey_data['Occupation'].unique())
    assert set(results.keys()) == expected_levels

def test_run_post_hoc_tukey_execution(tukey_data):
    """
    Smoke test to ensure the Tukey analysis completes without errors
    and produces summary objects.
    """
    try:
        results = run_post_hoc_tukey(tukey_data, 'Anxiety_Level', 'Gender', 'Occupation')
        # Check if the summary table for a specific level is not empty
        assert "Doctor" in results
        assert results["Doctor"] is not None
    except Exception as e:
        pytest.fail(f"Tukey Post-Hoc failed unexpectedly: {e}")

def test_run_post_hoc_tukey_error_handling():
    """
    Tests if the function handles empty DataFrames gracefully.
    Expected behavior: Returns an empty dictionary {} instead of None.
    """
    empty_df = pd.DataFrame(columns=['Anxiety_Level', 'Gender', 'Occupation'])
    results = run_post_hoc_tukey(empty_df, 'Anxiety_Level', 'Gender', 'Occupation')
    
    # Updated: Checking for an empty dictionary
    assert results == {}