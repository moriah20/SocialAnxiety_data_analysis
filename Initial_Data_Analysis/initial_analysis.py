import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from Visualization.visualization_saving_decorator import auto_save_plot


# -----------------------------
# Logger Configuration
# -----------------------------

logger = logging.getLogger(__name__)


# -----------------------------
# Null Check Function
# -----------------------------
def is_null(df):
    """
    Check whether the dataframe contains any null values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    bool
        True if null values exist, otherwise False.
    """
    try:
        null_count = df.isnull().sum(axis=0)
        has_nulls = null_count.sum() != 0
        logger.info(f"Null check completed. Contains nulls: {has_nulls}")
        return bool(has_nulls)
    except Exception as e:
        logger.error(f"Error in is_null: {e}")
        raise

def get_outliers_report(df, min_change_ratio=0.02):
    """
    Generate a detailed outlier report using the IQR method.
    Includes decision logic on whether outliers should be removed.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    min_change_ratio : float
        Minimum ratio of outlier rows required to justify removal.

    Returns
    -------
    outliers_detailed : pd.DataFrame
        Rows identified as outliers with reason.
    long_df : pd.DataFrame
        Long-format dataframe including bounds.
    should_remove : bool
        Whether the outlier removal is significant enough.
    """
    try:
        # Validate input
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a valid pandas DataFrame")

        # Select numeric columns only
        num_df = df.select_dtypes(include=['number'])
        if num_df.empty:
            raise ValueError("No numeric columns found for outlier detection")

        logger.info("Starting outlier detection using IQR method.")

        # Compute IQR bounds
        q1 = num_df.quantile(0.25)
        q3 = num_df.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Convert to long format for easier analysis
        long_df = num_df.reset_index().melt(
            id_vars='index',
            var_name='column',
            value_name='value'
        )

        # Attach bounds per column
        long_df['lower'] = long_df['column'].map(lower_bound)
        long_df['upper'] = long_df['column'].map(upper_bound)

        # Identify outliers
        outliers_detailed = long_df[
            (long_df['value'] < long_df['lower']) |
            (long_df['value'] > long_df['upper'])
        ].copy()

        # Add reason for each outlier
        outliers_detailed['reason'] = np.where(
            outliers_detailed['value'] > outliers_detailed['upper'],
            'Too High',
            'Too Low'
        )

        # Compute ratio of outliers
        total_rows = len(long_df)
        removed_rows = len(outliers_detailed)
        change_ratio = removed_rows / total_rows if total_rows > 0 else 0

        # Decide whether removal is justified
        should_remove = change_ratio >= min_change_ratio

        logger.info(
            f"Outlier detection completed. Found {removed_rows} outliers "
            f"({change_ratio:.2%} of data). "
            f"Removal decision: {'YES' if should_remove else 'NO'}."
        )

        return outliers_detailed, long_df, should_remove

    except Exception as e:
        logger.error(f"Error in get_outliers_report: {e}")
        raise

@auto_save_plot(output_dir="Visualization") #save and show plot
def plot_outliers(df):
    """
    Plot boxplots for raw and cleaned data after removing outliers.
    Uses decision logic from get_outliers_report to determine whether
    outlier removal is justified.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe or original dataframe if removal is not justified.
    """
    try:
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a valid pandas DataFrame")

        logger.info("Starting outlier plotting process.")

        # Get outlier report and decision flag
        outliers_df, long_df_raw, should_remove = get_outliers_report(df)

        # Helper function for plotting
        def draw(data, title):
            """Draws boxplots for each numeric column."""
            g = sns.FacetGrid(
                data,
                col="column",
                col_wrap=4,
                sharey=False,
                sharex=False,
                height=3,
                aspect=1.2
            )
            g.map(sns.boxplot, "value", color="skyblue")
            g.set_titles("{col_name}")
            g.set_xticklabels(rotation=45)

            g.fig.suptitle(title, fontsize=16) 
            g.fig.subplots_adjust(top=0.88)

        # Remove outliers
        long_df_clean = long_df_raw.drop(outliers_df.index)

        # If removal is not justified → plot only raw data
        if not should_remove:
            draw(long_df_raw, 'Raw Data (Outliers Detected but Not Removed)')
            draw(long_df_clean, 'Clean Data (After Removing Outliers)')
            logger.info("Outlier removal skipped due to low impact.")
            return df

        # Plot before/after
        draw(long_df_raw, 'Raw Data (With Outliers)')
        draw(long_df_clean, 'Clean Data (After Removing Outliers)')

        # Convert cleaned long-format back to wide format
        cleaned_df = long_df_clean.pivot(index='index', columns='column', values='value')
        cleaned_df = cleaned_df.reset_index(drop=True)

        logger.info("Outliers removed successfully. Returning cleaned dataframe.")
        return cleaned_df

    except Exception as e:
        logger.error(f"Error in plot_outliers: {e}")
        raise


@auto_save_plot(output_dir="Visualization") #save and show plot
# -----------------------------
# Frequency Plot for Categorical Variables
# -----------------------------
def frequency(df):
    """
    Plot frequency counts for all categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    None
    """
    try:
        cat_df = df.select_dtypes(include=['object', 'category'])

        if cat_df.empty:
            raise ValueError("No categorical columns found for frequency plotting")

        df_melted = cat_df.melt()

        g = sns.catplot(
            data=df_melted,
            kind="count",
            x="value",
            col="variable",
            col_wrap=2,
            sharex=False,
            sharey=False,
            palette="Paired"
        )

        g.set_xticklabels(rotation=45)
        g.fig.suptitle("Frequency plots", fontsize=16)
        g.fig.subplots_adjust(top=0.88, hspace=1.2, bottom=0.2, left=0.1, right=0.9)

        logger.info("Frequency plots generated successfully.")

    except Exception as e:
        logger.error(f"Error in frequency: {e}")
        raise

