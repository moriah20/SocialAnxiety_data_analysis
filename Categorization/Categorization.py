import pandas as pd
import logging

# Initialize module-level logger
logger = logging.getLogger(__name__)

import pandas as pd
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def categorization(binary_df, guideline):
    """
    Categorize binary variables according to a guideline table and compute
    the mean score (success rate) for each category.

    Parameters
    ----------
    binary_df : pd.DataFrame
        A dataframe containing binary columns (0/1) for each variable.
    guideline : pd.DataFrame
        A dataframe with two columns:
        - 'Category': category name
        - 'Variable': variable name that belongs to the category

    Returns
    -------
    pd.DataFrame
        A dataframe where each column represents a category and each row
        contains the average score for that category.
    """

    try:
        # Validate inputs
        if not isinstance(binary_df, pd.DataFrame):
            raise TypeError("binary_df must be a pandas DataFrame")

        if not isinstance(guideline, pd.DataFrame):
            raise TypeError("guideline must be a pandas DataFrame")

        if "Category" not in guideline.columns or "Variable" not in guideline.columns:
            raise ValueError("guideline must contain 'Category' and 'Variable' columns")

        logger.info("Starting categorization process")

        categorized_df = {}

        # Extract unique categories
        unique_categories = guideline["Category"].unique()

        for cat in unique_categories:
            # Get all variables belonging to the current category
            variables_in_cat = guideline[guideline["Category"] == cat]["Variable"].tolist()

            # Validate that all variables exist in the binary_df
            missing_vars = [v for v in variables_in_cat if v not in binary_df.columns]
            if missing_vars:
                raise KeyError(f"Variables missing in binary_df for category '{cat}': {missing_vars}")

            # Compute mean score for the category
            categorized_df[cat] = binary_df[variables_in_cat].sum(axis=1) / len(variables_in_cat)

        logger.info("Categorization completed successfully")
        return pd.DataFrame(categorized_df)

    except Exception as e:
        logger.error(f"Error during categorization: {e}")
        raise

