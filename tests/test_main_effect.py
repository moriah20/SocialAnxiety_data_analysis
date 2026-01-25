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

# --- Tests for check_effects (The Validator) ---

def test_check_effects_valid(anova_data):
    """Verifies the validator returns True for valid pandas Series."""
    # 1. Prepare the means (this is the step you were missing)
    m1 = anova_data.groupby('Gender')['Score'].mean()
    m2 = anova_data.groupby('Treatment')['Score'].mean()
    
    # 2. Assert the validation passes
    assert check_effects(m1, m2) == True

def test_main_effects_plots_execution(anova_data):
    """Ensures the plotting function runs without crashing (Smoke Test)."""
    # 1. Prepare the means
    m1 = anova_data.groupby('Gender')['Score'].mean()
    m2 = anova_data.groupby('Treatment')['Score'].mean()
    
    try:
        # 2. Run the plot function with the calculated means
        main_effects_plots(m1, m2, 'Gender', 'Treatment', 'Score')
        plt.close('all')
    except Exception as e:
        pytest.fail(f"Plotting failed even with valid Series: {e}")





