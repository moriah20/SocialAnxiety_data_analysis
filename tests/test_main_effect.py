import pytest
import pandas as pd
import matplotlib.pyplot as plt
from Occupation_Gender_Two_Way_ANOVA.main_effects import main_effects, check_effects, main_effects_plots

@pytest.fixture
def anova_data():
    return pd.DataFrame({
        'Gender': ['Male', 'Male', 'Female', 'Female'],
        'Treatment': ['GroupA', 'GroupB', 'GroupA', 'GroupB'],
        'Score': [90, 70, 85, 75]
    })

# --- Tests for main_effects ---
def test_main_effects_calculation(anova_data):
    """Verifies that means are calculated correctly for both IVs."""
    m1, m2 = main_effects(anova_data, 'Gender', 'Treatment', 'Score')
    
    # Check if results are Series
    assert isinstance(m1, pd.Series)
    assert isinstance(m2, pd.Series)
    
    # Check specific mean calculation (Female mean should be 80 in this dummy data)
    assert m1['Female'] == 80.0
    assert m2['GroupA'] == 87.5

def test_main_effects_invalid_input():
    """Should return (None, None) if input types are wrong."""
    m1, m2 = main_effects("Not a DF", 'IV1', 'IV2', 'DV')
    assert m1 is None and m2 is None


def test_main_effects_plots_execution(anova_data, tmp_path):
    m1 = anova_data.groupby('Gender')['Score'].mean()
    m2 = anova_data.groupby('Treatment')['Score'].mean()

    try:
        # Override output directory → ensures plots go to tmp_path
        main_effects_plots(
            m1, 
            m2, 
            'Gender', 
            'Treatment', 
            'Score',
            output_dir=tmp_path
        )

        # Verify that at least one plot was saved
        saved_files = list(tmp_path.glob("*.png"))
        assert len(saved_files) > 0, "No plot was saved to tmp_path"

        plt.close('all')

    except Exception as e:
        pytest.fail(f"Main effects plot failed: {e}")

