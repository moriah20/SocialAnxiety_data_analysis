
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from Occupation_Gender_Two_Way_ANOVA.interctions import calculate_interaction, plot_interaction_bar, run_post_hoc_tukey

@pytest.fixture
def interaction_data():
    """
    Generates a synthetic dataset designed to guarantee a statistical interaction.
    Logic: 
    - Males perform better in Treatment A than B.
    - Females perform better in Treatment B than A.
    This 'cross-over' effect is what creates the interaction.
    """
    return pd.DataFrame({
        'Gender': ['Male']*10 + ['Female']*10,
        'Treatment': (['A']*5 + ['B']*5) * 2,
        'Score': [
            90, 92, 88, 91, 89,  # Male A: High scores
            20, 22, 18, 21, 19,  # Male B: Low scores
            15, 17, 14, 16, 15,  # Female A: Low scores
            85, 87, 84, 86, 85   # Female B: High scores
        ]
    })

def test_calculate_interaction_success(interaction_data):
    """
    Tests if the interaction calculation returns valid results.
    Checks that both the p-value and the ANOVA model are not None.
    """
    # Unpack the results from the function
    p_val, model = calculate_interaction(interaction_data, 'Gender', 'Treatment', 'Score')
    
    # Assertions to verify successful calculation
    assert p_val is not None, "p-value should not be None"
    assert model is not None, "The statistical model should not be None"

def test_plot_interaction_runs_without_error(interaction_data):
    """
    Smoke test: Ensures the plotting function executes without crashing.
    Uses a try-except block to capture any runtime plotting errors.
    """
    try:
        plot_interaction_bar(interaction_data, 'Gender', 'Treatment', 'Score')
        # Close all figures to prevent memory issues and popup windows during tests
        plt.close('all') 
    except Exception as e:
        pytest.fail(f"Plotting function crashed. Error: {e}")




# --- Tests for calculate_interaction (The ANOVA wrapper) ---

def test_calculate_interaction_error_handling():
    """
    Tests if the function handles invalid input types.
    Expected behavior: Returns (None, None) for failed calculation.
    """
    # Passing a string instead of a DataFrame to trigger error handling
    results = calculate_interaction("Not a DataFrame", 'IV1', 'IV2', 'DV')
    
    # Updated: Checking for a tuple of two Nones
    assert results == (None, None)




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


def test_run_post_hoc_tukey_execution(interaction_data):
    """
    Tests if the Tukey post-hoc analysis executes correctly on interaction data.
    """
    # Running the post-hoc analysis
    results = run_post_hoc_tukey(interaction_data, 'Score', 'Gender', 'Treatment')
    
    # Verify that the results are returned (usually as a dictionary or DataFrame)
    assert results is not None
    assert len(results) > 0, "Tukey results should not be empty for this dataset"
def test_run_post_hoc_tukey_error_handling():
    """
    Tests if the function handles empty DataFrames gracefully.
    Expected behavior: Returns an empty dictionary {} instead of None.
    """
    empty_df = pd.DataFrame(columns=['Anxiety_Level', 'Gender', 'Occupation'])
    results = run_post_hoc_tukey(empty_df, 'Anxiety_Level', 'Gender', 'Occupation')
    
    # Updated: Checking for an empty dictionary
    assert results == {}