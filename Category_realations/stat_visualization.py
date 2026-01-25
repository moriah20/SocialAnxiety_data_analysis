import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import logging
from Visualization.visualization_saving_decorator import auto_save_plot


# Configure logger for the current module
logger = logging.getLogger(__name__)

@auto_save_plot(output_dir="Visualization") #save and show plot
def plot_spearman_bar_chart(spearman_df):
    """
    Plots a clean bar chart of Spearman correlation coefficients.
    Includes logging, error handling, and formatting updates for newer library versions.
    
    Args:
        spearman_df (pd.DataFrame): DataFrame containing 'Category' and 'Spearman Coeff'.
    """
    logger.info("Generating Spearman correlation bar chart...")

    try:
        # Validation: Check if DataFrame is empty to prevent errors
        if spearman_df.empty:
            logger.warning("Spearman DataFrame is empty. Skipping plot.")
            return

        # Initialize the figure size
        plt.figure(figsize=(10, 6))
        
        # Define color scheme (Pastel Blue, Gray, Orange)
        custom_colors = ['#8da0cb', '#d9d9d9', '#fc8d62'] 
        
        # Create the bar plot
        # Note: 'hue' and 'legend=False' are added to comply with future Seaborn versions
        ax = sns.barplot(
            x='Category', 
            y='Spearman Coeff', 
            data=spearman_df, 
            palette=custom_colors,
            hue='Category',    # Explicitly assigning hue to avoid deprecation warning
            legend=False,      # Disable legend as it's redundant here
            edgecolor='black', # Add thin border for better visibility
            linewidth=0.5
        )

        # Set titles and axis labels
        plt.title('Spearman Correlation Coefficients', fontsize=14, pad=20)
        plt.ylabel('Correlation Coefficient (r)', fontsize=12)
        plt.xlabel('Category', fontsize=12)

        # Annotate bars with numerical values
        for p in ax.patches:
            height = p.get_height()
            # Calculate offset to place text slightly above/below the bar
            offset = -0.02 if height < 0 else 0.01
            
            ax.text(
                p.get_x() + p.get_width() / 2.,  # X position (center of bar)
                height + offset,                 # Y position
                f'{height:.2f}',                 # Format to 2 decimal places
                ha="center", 
                va="top" if height < 0 else "bottom",
                fontsize=10,
                color='black'
            )

        # Add a reference line at zero and format axes
        plt.axhline(0, color='black', linewidth=0.8) 
        plt.xticks(rotation=45) # Rotate labels for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.3) 
        plt.tight_layout()
        
        # Display the plot
        logger.info("Displaying Spearman plot window.")

    except Exception as e:
        logger.exception("An error occurred while plotting the Spearman bar chart.")
        raise e


@auto_save_plot(output_dir="Visualization") #save and show plot
def plot_regression_summary(model):
    """
    Visualizes the regression coefficients with their 95% Confidence Intervals.
    
    Args:
        model: The fitted statsmodels OLS result object.
    """
    logger.info("Generating Regression Coefficients plot...")

    try:
        # 1. Extract coefficients and confidence intervals from the model
        results_df = pd.DataFrame({
            'coef': model.params,
            'lower_ci': model.conf_int()[0],
            'upper_ci': model.conf_int()[1]
        })
        
        # 2. Remove the Constant (Intercept) from the plot
        # The intercept scale is often different from predictors, so we exclude it.
        if 'const' in results_df.index:
            results_df = results_df.drop('const')
            logger.debug("Intercept (const) removed from visualization data.")

        # 3. Calculate error bar sizes (distance from coefficient to CI bounds)
        yerr_lower = results_df['coef'] - results_df['lower_ci']
        yerr_upper = results_df['upper_ci'] - results_df['coef']
        
        # 4. Create the plot
        plt.figure(figsize=(10, 6))

        # Plot coefficients as points with error bars
        plt.errorbar(
            x=results_df.index.astype(str).tolist(), 
            y=results_df['coef'], 
            yerr=[yerr_lower, yerr_upper], 
            fmt='o',            # Circle marker
            color='#c0392b',    # Dark red color
            ecolor='black',     # Black error bars
            capsize=5,          
            elinewidth=2,       
            markersize=8
        )
        
        # 5. Add Reference Line at Y=0 (Significance threshold)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        # 6. Labels and Title
        plt.title('Regression Coefficients: Impact on Anxiety Level', fontsize=14, fontweight='bold')
        plt.ylabel('Coefficient Value (Change in Anxiety)', fontsize=12)
        plt.xlabel('Predictor Categories', fontsize=12)
        
        # 7. Add text annotations for exact values
        for i, txt in enumerate(results_df['coef']):
            plt.annotate(
                f"{txt:.2f}", 
                # FIXED: Ensure the x-coordinate is passed as a string to match the list above
                (str(results_df.index[i]), results_df['coef'].iloc[i]), 
                xytext=(15, 0), 
                textcoords='offset points',
                fontsize=10, 
                fontweight='bold'
            )

        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        logger.info("Displaying Regression plot window.")

    except Exception as e:
        logger.exception("An error occurred while plotting the Regression summary.")
        raise e