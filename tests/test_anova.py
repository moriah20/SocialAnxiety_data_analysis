import pytest
import pandas as pd
import numpy as np
from Two_Way_ANOVA.two_way_anova import two_way_anova_test, run_post_hoc_analysis 

@pytest.fixture
def sample_data():
    """Generates a valid dummy dataset for testing."""
    return pd.DataFrame({
        'Gender': ['Male', 'Male', 'Female', 'Female'] * 5,
        'Employment': ['Yes', 'No', 'Yes', 'No'] * 5,
        'Score': np.random.randint(10, 100, size=20)
    })

@pytest.fixture
def empty_df():
    """Generates an empty dataframe."""
    return pd.DataFrame(columns=['Score', 'Gender', 'Employment'])

# Test 1: Check if it handles empty DataFrames gracefully
def test_two_way_anova_empty_df(empty_df):
    """Should return None and log an error when df is empty."""
    result = two_way_anova_test(empty_df, "Score", "Gender", "Employment", "Score_Clean")
    assert result is None

# Test 3: Valid full run
def test_two_way_anova_success(sample_data):
    """Should return a pandas DataFrame (the ANOVA table) on success."""
    result = two_way_anova_test(sample_data, "Score", "Gender", "Employment", "Score_Clean")
    assert isinstance(result, pd.DataFrame)
    assert 'Source' in result.columns
    assert 'p-unc' in result.columns

# Test 4: Post-hoc analysis logic
def test_run_post_hoc_success(sample_data):
    """Should return pairwise comparison results."""
    # We must prepare the column first as the main function does
    sample_data['Score_Clean'] = pd.to_numeric(sample_data['Score'], errors='coerce')
    result = run_post_hoc_analysis(sample_data, "Score_Clean", "Gender", "Employment")
    assert isinstance(result, pd.DataFrame)
    assert 'p-corr' in result.columns

# Test 2: Updated to expect None if column is missing (assuming you fix the function)
def test_two_way_anova_missing_column(sample_data):
    """Should return None if a specified column does not exist."""
    # The function should internally handle this or the test should expect None
    result = two_way_anova_test(sample_data, "Wrong_Column", "Gender", "Employment", "Score_Clean")
    assert result is None

# Test 5: Updated with more rows to satisfy pingouin's minimum requirement (N>=5)
def test_non_numeric_data():
    """Should handle strings in numeric columns via coerce and have enough data for ANOVA."""
    df = pd.DataFrame({
        'Gender': ['M', 'M', 'F', 'F', 'M', 'F', 'M'], # 7 rows
        'Employment': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y'],
        'Score': ['10', '20', 'invalid', '40', '30', '50', '60']
    })
    result = two_way_anova_test(df, "Score", "Gender", "Employment", "Score_Clean")
    
    # After dropping 'invalid', we have 6 valid rows. ANOVA requires >= 5.
    assert result is not None
    assert isinstance(result, pd.DataFrame)