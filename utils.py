
import pandas as pd
import numpy as np
import warnings

def reduce_dimension(metadata, data, percent):
    """
    Reduces the dimension of data by a certain percent for each chromosome.

    Args:
    metadata (pd.DataFrame): A dataframe containing the metadata for chromosomes. 
                             It must have the columns "chr", "start", and "end".
    data (pd.DataFrame): The adjacency matrix represented as a dataframe, representing contacts between bins.
    percent (float): The percentage by which to reduce the data.

    Returns:
    pd.DataFrame: The reduced dataframe.
    pd.DataFrame: The updated metadata dataframe.
    """
    # Assertion to ensure that the percent is a valid value.
    if percent < 0.0 or percent > 1.0:
        raise ValueError("Percent should be between 0 and 1 (inclusive).")
    if percent == 1.0:
        # If percent is 1, all data and metadata should be dropped.
        warnings.warn("The resulting data and metadata are empty because percent equals 1.")
        return pd.DataFrame(), pd.DataFrame()

    new_data = data.copy(deep=True)
    new_metadata = metadata.copy(deep=True)
    
    num_bins_tot = 0
    # Iterate over each chromosome in the metadata
    for chr, row in new_metadata.iterrows():
      # Calculate the number of bins to remove
      num_bins = int(np.ceil((row['end'] - row['start'] + 1) * percent))

      # The bins to remove are the last num_bins bins
      bins_to_remove = np.arange(row['end'] - num_bins + 1, row['end'] + 1)

      # Remove the bins from the data
      new_data.drop(bins_to_remove, errors='ignore', inplace=True)
      new_data.drop(bins_to_remove, axis=1, errors='ignore', inplace=True)
      
      num_bins_tot += num_bins
      # Update the end index in the metadata
      new_metadata.loc[chr, 'end'] -= num_bins_tot

    # After dropping bins from the data, update the start indices in the metadata
    for chr in new_metadata.index[1:]:
        new_metadata.loc[chr, 'start'] = new_metadata.loc[chr-1, 'end'] + 1
    
    # Reset the index and the columns of the data 
    new_data = new_data.reset_index(drop=True)
    new_data.columns = pd.RangeIndex(len(new_data.columns))

    return new_data, new_metadata
        
