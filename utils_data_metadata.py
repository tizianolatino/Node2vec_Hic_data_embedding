#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 09:17:31 2023

@author: tizianolatino
"""

def adjust_indices_to_zero(metadata, data):
    """
    Adjust the 'start' and 'end' points in the metadata, and the indices in the data, to start from 0.

    Args:
    metadata (pd.DataFrame): A dataframe containing the metadata for chromosomes. 
                             It must have the columns "chr", "start", and "end".
    data (pd.DataFrame): The adjacency matrix represented as a dataframe, representing contacts between bins.

    Returns:
    pd.DataFrame, pd.DataFrame: The updated metadata and data dataframes.
    """
    # Check if the first 'start' point in the metadata is 0
    if metadata.iloc[0]['start'] != 0:
        adjustment = metadata.iloc[0]['start']
        # Adjust the 'start' and 'end' points in the metadata
        metadata['start'] -= adjustment
        metadata['end'] -= adjustment
    
    # Check if the first index in the data is 0
    if data.index[0] != 0:
        # Reset the index of the data to start from 0
        data = data.reset_index(drop=True)
    
    return metadata, data
