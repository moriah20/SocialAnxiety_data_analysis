import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import your functions here.
# Assuming your file is named 'main.py'
from main import is_null, get_outliers_report, plot_outliers, frequency

class TestDataAnalysis(unittest.TestCase):

    def setUp(self):
        """
        This function runs before every test.
        We create small DataFrames here for testing to avoid dependency on an external CSV file.
        """
        # 1. Data without missing values
        self.clean_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })

        # 2. Data with missing values (NaN)
        self.dirty_df = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': ['x', 'y', None]
        })

        # 3. Data with clear outliers
        # In the range 1-5, the number 100 is a clear outlier
        self.outlier_df = pd.DataFrame({
            'Age': [20, 21, 19, 22, 100, 20], 
            'Salary': [5000, 5100, 4900, 5050, 5000, 200] # 200 is also a low outlier
        })

        # 4. Categorical data for testing frequency function
        self.cat_df = pd.DataFrame({
            'Gender': ['Male', 'Female', 'Female', 'Male'],
            'City': ['TA', 'TA', 'Jerusalem', 'Haifa']
        })

    # --- Tests for is_null function ---

    def test_is_null_with_missing_values(self):
        """Test that the function correctly identifies missing values."""
        result = is_null(self.dirty_df)
        self.assertTrue(result, "Should return True when DataFrame contains nulls")

    def test_is_null_clean(self):
        """Test that the function returns False when everything is clean."""
        result = is_null(self.clean_df)
        self.assertFalse(result, "Should return False when DataFrame has no nulls")

    # --- Tests for get_outliers_report function ---

    def test_get_outliers_logic(self):
        """Test that the function correctly identifies mathematical outliers."""
        outliers, long_df = get_outliers_report(self.outlier_df)
        
        # We expect the value 100 in the 'Age' column to be identified as an outlier
        age_outliers = outliers[outliers['column'] == 'Age']
        self.assertIn(100, age_outliers['value'].values)
        
        # We expect the value 100 to be marked as "Too High"
        reason = age_outliers[age_outliers['value'] == 100]['reason'].iloc[0]
        self.assertEqual(reason, 'Too High')

    # --- Tests for plot_outliers function ---

    @patch('matplotlib.pyplot.show') # Mocks/Disables the graph display
    def test_plot_outliers_removes_data(self, mock_show):
        """Test that the function returns clean data (fewer rows than the original)."""
        
        # The original data contains 6 rows (including outliers)
        initial_count = len(self.outlier_df)
        
        # Run the function
        cleaned_df = plot_outliers(self.outlier_df)
        
        # The new data should contain fewer rows
        final_count = len(cleaned_df)
        self.assertLess(final_count, initial_count, "Cleaned DF should be smaller than original")
        
        # Ensure the outlier (100) is no longer in the clean data
        self.assertNotIn(100, cleaned_df['Age'].values)

    # --- Tests for frequency function ---

    @patch('matplotlib.pyplot.show') # Mocks the graph display
    @patch('seaborn.catplot')        # Mocks the heavy seaborn plotting
    def test_frequency_runs_successfully(self, mock_catplot, mock_show):
        """Test that the function runs without crashing on categorical data."""
        try:
            frequency(self.cat_df)
        except Exception as e:
            self.fail(f"frequency() raised an exception unexpectedly: {e}")
            
        # Verify that the function actually tried to plot something
        mock_catplot.assert_called() 

if __name__ == '__main__':
    unittest.main()