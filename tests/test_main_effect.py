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
    """Should return True for valid pandas Series."""
    s1 = anova_data.groupby('Gender')['Score'].mean()
    s2 = anova_data.groupby('Treatment')['Score'].mean()
    assert check_effects(s1, s2) is True

def test_check_effects_non_numeric():
    """Should return False if Series contains non-numeric data."""
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series([1, 2])
    assert check_effects(s1, s2) is False

def test_main_effects_plots_execution(anova_data):
    """
    Smoke test: Verifies that the plotting function runs without crashing.
    This covers Matplotlib/Seaborn integration.
    """
    # Prepare valid inputs using your existing functions
    m1, m2 = main_effects(anova_data, 'Gender', 'Treatment', 'Score')
    
    try:
        # Run the plot function
        main_effects_plots(m1, m2, 'Gender', 'Treatment', 'Score')
        # Close all figures to prevent memory leaks during tests
        plt.close('all') 
    except Exception as e:
        pytest.fail(f"Plotting function crashed unexpectedly! Error: {e}")

