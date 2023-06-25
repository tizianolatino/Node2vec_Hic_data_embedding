#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:46:56 2023

@author: tizianolatino
"""

import numpy as np
import pandas as pd


def drop_chr(metadata, data, chr_to_drop):
    """
    Removes a specified chromosome from the metadata dataframe, removes corresponding
    bins from the data adjacency matrix, and adjusts the start and end points 
    of the remaining chromosomes in the metadata.

    Args:
    metadata (pd.DataFrame): A dataframe containing the metadata for chromosomes. 
                             It must have the columns "chr", "start", and "end".
    data (pd.DataFrame): The adjacency matrix represented as a dataframe, representing contacts between bins.
    chr_to_drop (str): The name of the chromosome to remove.

    Returns:
    tuple: A tuple containing the new metadata dataframe with the specified chromosome 
           removed and the start and end points of the remaining chromosomes adjusted, 
           and the new adjacency matrix with the bins of the dropped chromosome removed.

    Raises:
    ValueError: If no chromosome with the provided name is found in the metadata.

    """
   
    
    # Find the row for the chromosome to drop
    chr_row = metadata.loc[metadata['chr'] == chr_to_drop]
    
    if chr_row.empty:
        raise ValueError(f"No chromosome found with name {chr_to_drop}")
        
    # Calculate the length of the chromosome
    chr_length = chr_row.iloc[0]['end'] - chr_row.iloc[0]['start'] + 1
    
    # Drop the bins of the chromosome from the adjacency matrix
    data = data.drop(labels=range(chr_row.iloc[0]['start'], chr_row.iloc[0]['end'] + 1), axis=0)
    data = data.drop(labels=range(chr_row.iloc[0]['start'], chr_row.iloc[0]['end'] + 1), axis=1)
    
    # Drop the chromosome from the metadata
    metadata = metadata[metadata['chr'] != chr_to_drop]
    
    # Adjust the start and end points of the remaining chromosomes
    metadata.loc[metadata['start'] > chr_row.iloc[0]['start'], 'start'] -= chr_length
    metadata.loc[metadata['end'] > chr_row.iloc[0]['end'], 'end'] -= chr_length
    
    # Reset the index and the columns of the data 
    data = data.reset_index(drop=True)
    data.columns = pd.RangeIndex(len(data.columns))
    
    # Reset the index and the columns of the metadata
    metadata = metadata.reset_index(drop=True)
    
    return metadata, data

def remove_isolated_bins(metadata, data):
    """
    Removes isolated bins (with zero contacts) from the data adjacency matrix 
    and adjusts the start and end points of the chromosomes in the metadata dataframe accordingly.

    Args:
    metadata (pd.DataFrame): A dataframe containing the metadata for chromosomes. 
                             It must have the columns "chr", "start", and "end".
    data (pd.DataFrame): The adjacency matrix represented as a dataframe, representing contacts between bins.

    Returns:
    tuple: A tuple containing the new metadata dataframe with the start and end points 
           of the chromosomes adjusted, and the new adjacency matrix with the isolated bins removed.
    """
    
    
    # Find the rows and columns in the adjacency matrix that are all zero
    isolated_bins = data.index[(data == 0).all(1)]
    
    # Drop the isolated bins from the adjacency matrix
    data = data.drop(isolated_bins, axis=0)
    data = data.drop(isolated_bins, axis=1)
    
    # Reset index of the metadata dataframe
    data = data.reset_index(drop=True)
    # Reset column index
    data.columns = pd.RangeIndex(len(data.columns))
    
    # Iterate over each chromosome in the metadata
    for i, row in metadata.iterrows():
        
        # Count how many isolated bins are in this chromosome
        isolated_bins_in_chr = isolated_bins[(isolated_bins >= row['start']) & (isolated_bins <= row['end'])]

        bin_count = len(isolated_bins_in_chr)

        if bin_count > 0:
           # If there are isolated bins, adjust the 'end' point for this chromosome
           metadata.at[i, 'end'] -= bin_count
           
        # Adjust the 'start' and 'end' points for all following chromosomes
        metadata.loc[metadata.index > i, 'start'] -= bin_count
        metadata.loc[metadata.index > i, 'end'] -= bin_count


    return metadata, data
    



