
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from Occupation_Gender_Two_Way_ANOVA.interctions import calculate_interaction, plot_interaction_bar, run_post_hoc_tukey


@pytest.fixture
def interaction_data():
    """Generates a synthetic dataset designed to guarantee a statistical interaction."""
    return pd.DataFrame({
        'Gender': ['Male']*10 + ['Female']*10,
        'Treatment': (['A']*5 + ['B']*5) * 2,
        'Score': [
            90, 92, 88, 91, 89,  # Male A: High scores
            20, 22, 18, 21, 19,  # Male B: Low scores
            15, 17, 14, 16, 15,  # Female A: Low scores
            85, 87, 4, 86, 85   # Female B: High scores
        ]
    })

def test_calculate_interaction_success(interaction_data):
    p_val, model = calculate_interaction(interaction_data, 'Gender', 'Treatment', 'Score')
    assert p_val is not None
    assert model is not None


def test_plot_interaction_runs_without_error(interaction_data):
    try:
        # Added the missing 'Score' argument
        plot_interaction_bar(interaction_data, 'Gender', 'Treatment', 'Score')
        plt.close('all')
    except Exception as e:
        pytest.fail(f"Plotting crashed: {e}")

def tukey_data():
    """Generates synthetic data for Tukey HSD testing."""
    return pd.DataFrame({
        'Anxiety_Level': [3.0, 3.2, 5.0, 5.2, 2.0, 2.1, 4.0, 4.3],
        'Gender': ['Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female'],
        'Occupation': ['Doctor', 'Doctor', 'Doctor', 'Doctor', 'Artist', 'Artist', 'Artist', 'Artist']
    })

def test_run_post_hoc_tukey_structure(interaction_data):
    results = run_post_hoc_tukey(interaction_data, 'Score', 'Gender', 'Treatment')
    # Your function returns a dict, so we check for that
    assert isinstance(results, dict)
    assert len(results) > 0




def test_run_post_hoc_tukey_error_handling():
    """Tests that the function returns None on error (as per your code's 'return None')."""
    empty_df = pd.DataFrame(columns=['Score', 'Gender', 'Treatment'])
    results = run_post_hoc_tukey(empty_df, 'Score', 'Gender', 'Treatment')
    
    # Since your function has 'return None' in the except block
    assert results is None
    
def test_run_post_hoc_tukey_execution(interaction_data):
    """
    Tests if the Tukey post-hoc analysis executes correctly on interaction data.
    """
    # Running the post-hoc analysis
    results = run_post_hoc_tukey(interaction_data, 'Score', 'Gender', 'Treatment')
    
    # Verify that the results are returned (usually as a dictionary or DataFrame)
    assert results is not None
    assert len(results) > 0, "Tukey results should not be empty for this dataset"










