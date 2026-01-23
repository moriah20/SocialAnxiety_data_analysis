import pandas as pd
import logging

# Initialize module-level logger
logger = logging.getLogger(__name__)

def apply_binary(df, guidelines):
    """
    Transforms raw columns into binary format (0/1) based on mapping and numeric range guidelines.

    Parameters
    ----------
    df : pd.DataFrame
        The source dataframe containing the raw data to be processed.
    guidelines : pd.DataFrame
        A configuration dataframe with the following required columns:
        - 'Variable': Name of the column in `df` to process.
        - 'Type': The logic to apply ('binary' or 'numeric').
        - 'Condition': If 'no', binary values are reversed (0 becomes 1 and vice versa).
        - 'Min' / 'Max': The inclusive range used to binarize 'numeric' types.

    Returns
    -------
    pd.DataFrame
        A filtered dataframe containing only the variables listed in the guidelines, 
        where all values have been converted to integers (0 or 1).

    """
    # Define constants 
    BINARY_TYPE = "binary"
    NUMERIC_TYPE = "numeric"
    MAPPING = {"yes": 1, "no": 0}
    REVERSE_MAPPING = {0: 1, 1: 0}
    
    # Avoid modifying the original dataframe
    processed_df = df.copy() 
    
    try:
        logger.info("Starting data binarization process.")
        
        for i, category in guidelines["Variable"].items():
            # Validate if the column exists in the dataset
            if category not in processed_df.columns:
                logger.error(f"Alignment Error: Column '{category}' missing from dataset.")
                raise KeyError(f"Column '{category}' not found in dataset.")
            
            var_type = str(guidelines["Type"][i]).lower()

            # Logic for binary classification
            if var_type == BINARY_TYPE:
                processed_df[category] = processed_df[category].astype(str).str.strip().str.lower()
                processed_df[category] = processed_df[category].replace(MAPPING)
                
                # Reverse mapping if the condition is "no"
                if str(guidelines["Condition"][i]).lower() == "no":
                    processed_df[category] = processed_df[category].replace(REVERSE_MAPPING)
                
                logger.debug(f"Successfully processed binary column: {category}")

            # Logic for numeric range binarization
            elif var_type == NUMERIC_TYPE:
                min_val, max_val = guidelines["Min"][i], guidelines["Max"][i]
                processed_df[category] = processed_df[category].between(min_val, max_val).astype(int)
                logger.debug(f"Successfully processed numeric column: {category}")
            
            else:
                # If the type in Excel is neither binary nor numeric
                raise ValueError(f"Unknown variable type '{var_type}' for category '{category}'.")

        # Extract only relevant variables
        target_cols = list(guidelines["Variable"])
        logger.info(f"Transformation complete. Processed {len(target_cols)} variables.")
        return processed_df[target_cols]

    except (KeyError, ValueError) as e:
        # Log specific data issues 
        logger.error(f"Data Validation Error: {e}")
        raise #stop run
    except Exception as e:
        # Log any other unexpected system errors
        logger.exception("Unexpected error during transformation stage.")
        raise
    

