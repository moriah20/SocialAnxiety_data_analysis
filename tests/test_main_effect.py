import pytest
import pandas as pd
import matplotlib.pyplot as plt
from Occupation_Gender_Two_Way_ANOVA.main_effects import main_effects, check_effects, main_effects_plots

@pytest.fixture
# --- Tests for main_effects ---

def test_main_effects_calculation(anova_data):
    """Verifies that means are calculated correctly for both IVs."""
    m1, m2 = main_effects(anova_data, 'Gender', 'Treatment', 'Score')
    
    # Check if results are Series
    assert isinstance(m1, pd.Series)
    assert isinstance(m2, pd.Series)
    
    # Check specific mean calculation (Female mean should be 80 in this dummy data)
    assert m1['Female'] == 80.0
    assert m2['GroupA'] == 82.5

def test_main_effects_invalid_input():
    """Should return (None, None) if input types are wrong."""
    m1, m2 = main_effects("Not a DF", 'IV1', 'IV2', 'DV')
    assert m1 is None and m2 is None



def test_check_effects_valid(anova_data):
    # Convert DataFrame to the Series the function expects
    m1 = anova_data.groupby('Gender')['Score'].mean()
    m2 = anova_data.groupby('Treatment')['Score'].mean()
    assert check_effects(m1, m2) == True

def test_main_effects_plots_execution(anova_data):
    m1 = anova_data.groupby('Gender')['Score'].mean()
    m2 = anova_data.groupby('Treatment')['Score'].mean()
    try:
        # Passing all 5 required arguments
        main_effects_plots(m1, m2, 'Gender', 'Treatment', 'Score')
        plt.close('all')
    except Exception as e:
        pytest.fail(f"Main effects plot failed: {e}")

