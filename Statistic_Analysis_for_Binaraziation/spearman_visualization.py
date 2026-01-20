import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_correlation_heatmap(results_df):
    """
    Plots a heatmap of the Spearman correlation coefficients.
    Adds a star (*) annotation if the result is statistically significant (p < 0.05).
    
    Args:
        results_df: The DataFrame returned by the spearman_test function.
    """
    
    # Create a copy to avoid modifying the original data
    df_plot = results_df.copy()

    # Sort the results by correlation coefficient for better visualization
    df_plot = df_plot.sort_values(by='Spearman Coeff', ascending=False)

    # Set the 'Category' as the index (so it appears on the Y-axis)
    # We only want the 'Spearman Coeff' column for the heatmap color
    heatmap_data = df_plot.set_index('Category')[['Spearman Coeff']]

    # Create labels for the heatmap cells
    # Format: "0.123" or "0.123 *" if p-value < 0.05
    labels = []
    for index, row in df_plot.iterrows():
        label = f"{row['Spearman Coeff']:.2f}"
        if row['P-value'] < 0.05:
            label += " *"  # Add star for significance
        labels.append(label)
    
    # Reshape labels to match the heatmap data shape (n_rows, 1_column)
    labels_matrix = np.array(labels).reshape(-1, 1)

    # Set up the matplotlib figure size
    plt.figure(figsize=(8, len(df_plot) * 0.8)) # Adjust height based on number of categories

    # Draw the heatmap
    sns.heatmap(
        heatmap_data, 
        annot=labels_matrix,  # Use our custom labels with stars
        fmt='',               # Tell seaborn that annotations are raw strings
        cmap='coolwarm',      # Color map: Blue (negative) -> Red (positive)
        center=0,             # Center the color map at 0
        vmin=-1, vmax=1,      # Fix the scale from -1 to 1
        cbar_kws={'label': 'Spearman Coefficient'}, # Label for the color bar
        linewidths=.5         # Add lines between cells
    )

    plt.title('Correlation with Anxiety Level', fontsize=16)
    plt.ylabel('') # Remove Y-axis label as categories are self-explanatory
    plt.tight_layout()
    plt.show()