import pytest
import pandas as pd
import matplotlib.pyplot as plt
from Occupation_Gender_Two_Way_ANOVA.interctions import calculate_interaction, plot_interaction_bar, run_post_hoc_tukey


@pytest.fixture
def interaction_data():
    """Generates a synthetic dataset designed to guarantee a statistical interaction."""
    
    return  pd.DataFrame({
        'Gender': ['Male']*10 + ['Female']*10,
        'Treatment': (['A']*5 + ['B']*5) * 2,
        'Score': [
            90, 92, 88, 91, 89,  # Male A: High scores
            20, 22, 18, 21, 19,  # Male B: Low scores
            15, 17, 14, 16, 15,  # Female A: Low scores
            85, 87, 84, 86, 85   # Female B: High scores
        ]
    })

def p_val():
    p_val=0.05
    return p_val


def test_calculate_interaction_success(interaction_data):
    """
    Test if calculate_interaction correctly returns three values 
    and successfully processes valid data.
    """
    # Act: Perform the calculation
    # Now expecting 3 return values: results, clean_df, and p_val
    results, clean_df, p_val = calculate_interaction(interaction_data, 'Score', 'Gender', 'Treatment')
    
    # Assert: Verify outputs are not None
    assert results is not None, "ANOVA results table should not be None"
    assert clean_df is not None, "Cleaned DataFrame should not be None"
    
    # Verify that p_val is either a number (float/int) or None (if no interaction found)
    assert p_val is None or isinstance(p_val, (float, int)), "p_val should be a numeric type or None"

def test_plot_interaction_runs_without_error(interaction_data, tmp_path):
    """
    Test if the plotting function executes without crashing (smoke test),
    and ensure plots are saved only to a temporary directory.
    """
    try:
        # Arrange
        res, clean_df, p_val = calculate_interaction(interaction_data, 'Score', 'Gender', 'Treatment')

        # Act — override output directory to tmp_path
        plot_interaction_bar(
            clean_df,
            'Gender',
            'Treatment',
            'Score',
            p_val,
            output_dir=tmp_path
        )

        # Assert — check that at least one file was saved
        saved_files = list(tmp_path.glob("*.png"))
        assert len(saved_files) > 0, "No plot was saved to tmp_path"

        # Cleanup
        plt.close('all')

    except Exception as e:
        pytest.fail(f"Plotting function crashed unexpectedly: {e}")

def test_calculate_interaction_insufficient_data():
    """
    Test if the function handles cases with very few rows gracefully.
    """
    # Arrange: Create a tiny dataframe (less than 5 rows)
    small_df = pd.DataFrame({
        'Score': [1, 2],
        'Gender': ['M', 'F'],
        'Treatment': ['A', 'B']
    })
    
    # Act: Run the function
    results, clean_df, p_val = calculate_interaction(small_df, 'Score', 'Gender', 'Treatment')
    
    # Assert: Should return (None, None, None) as per our error handling logic
    assert results is None
    assert clean_df is None
    assert p_val is None

    
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










