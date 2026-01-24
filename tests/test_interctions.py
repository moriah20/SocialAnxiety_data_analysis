import pytest
import pandas as pd
import numpy as np
from Occupation_Gender_Two_Way_ANOVA.interctions import main_effects, check_effects, calculate_interaction # Replace 'main' with your filename

@pytest.fixture
def anova_data():
    """Generates a valid dummy dataset for ANOVA testing."""
    return pd.DataFrame({
        'Gender': ['Male', 'Male', 'Female', 'Female'] * 5,
        'Treatment': ['GroupA', 'GroupB', 'GroupA', 'GroupB'] * 5,
        'Score': [80, 70, 85, 75] * 5
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

# --- Tests for calculate_interaction (The ANOVA wrapper) ---

def test_calculate_interaction_success(anova_data):
    """Verifies that pg.anova is called and returns a results table."""
    results = calculate_interaction(anova_data, 'Score', 'Gender', 'Treatment')
    assert isinstance(results, pd.DataFrame)
    assert 'Source' in results.columns
    assert 'p-unc' in results.columns


def test_calculate_interaction_error_handling(anova_data):
    """Should return None if column names are incorrect."""
    # Ensure this returns None instead of raising KeyError
    results = calculate_interaction(anova_data, 'Wrong_Column', 'Gender', 'Treatment')
    assert results is None