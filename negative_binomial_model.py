#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:43:09 2023

@author: tizianolatino
"""

import pandas as pd
import numpy as np
from scipy.stats import nbinom

def intrachr_contacts_mean_var(metadata, data):
    """
    Calculates the mean and variance of contact value for each distance within each chromosome.

    Args:
    metadata (pd.DataFrame): A dataframe containing the metadata for chromosomes. 
                             It must have the columns "chr", "start", and "end".
    data (pd.DataFrame): The adjacency matrix represented as a dataframe, representing contacts between bins.

    Returns:
    dict: Two dictionaries where the keys are the chromosomes, each item is a dictionary where each key is a distance,
          and the value is the mean and variance of contact value for that distance within the chromosome.
    """
    distance_means = {}
    distance_vars = {}

    # Iterate over each chromosome in the metadata
    for _, row in metadata.iterrows():
        # Select the data for this chromosome
        chr_data = data.iloc[row['start']:row['end']+1, row['start']:row['end']+1]

        # Calculate the distance for each pair of bins
        distances = np.abs(np.subtract.outer(np.arange(row['end'] - row['start'] + 1), np.arange(row['end'] - row['start'] + 1)))

        # Flatten the distances and the data, and create a DataFrame
        df = pd.DataFrame({'distance': distances.flatten(), 'value': chr_data.values.flatten()})

        # Group by distance and calculate the mean and variance
        distance_stats = df.groupby('distance')['value'].agg(['mean', 'var'])

        # Store the results into the dictionary
        distance_means[row['chr']] = distance_stats['mean'].to_dict()
        distance_vars[row['chr']] = distance_stats['var'].to_dict()

    return distance_means, distance_vars


def interchr_contacts_mean_var(metadata, data):
    """
   Calculates the mean and variance of contact value for interchromosomal interactions.

   Args:
   metadata (pd.DataFrame): A dataframe containing the metadata for chromosomes. 
                            It must have the columns "chr", "start", and "end".
   data (pd.DataFrame): The adjacency matrix represented as a dataframe, representing contacts between bins.

   Returns:
   tuple: The mean and variance of contact value for interchromosomal interactions.
   """
    # Create a copy of the data to avoid modifying the original
    data_copy = data.copy()

    # Set the intrachromosomal elements to NaN
    for _, row in metadata.iterrows():
        data_copy.iloc[row['start']:row['end']+1, row['start']:row['end']+1] = np.nan

    # The remaining elements are the interchromosomal interactions
    interchromosomal_values = data_copy.values.flatten()

    # Remove NaN values
    interchromosomal_values = interchromosomal_values[~np.isnan(interchromosomal_values)]

      # Calculate the mean and variance
    mean = interchromosomal_values.mean()
    var = interchromosomal_values.var()

    return mean, var

def generate_data_from_neg_binomial(metadata, data):
    """
    Generates a new dataframe with the same shape as data, using a negative binomial distribution.

    Args:
    metadata (pd.DataFrame): A dataframe containing the metadata for chromosomes. 
                             It must have the columns "chr", "start", and "end".
    data (pd.DataFrame): The adjacency matrix represented as a dataframe, representing contacts between bins.

    Returns:
    pd.DataFrame: A new dataframe with values generated from a negative binomial distribution.
    """
    # Calculate the mean and variance for each distance within each chromosome, 
    # and for interchromosomal interactions
    distance_means, distance_vars = intrachr_contacts_mean_var(metadata, data)
    interchr_mean, interchr_var = interchr_contacts_mean_var(metadata, data)

    # Initialize the new data with zeros
    new_data = pd.DataFrame(0, index=data.index, columns=data.columns)
    
    
    # For each chromosome
    for _, row in metadata.iterrows():
        prev_p = prev_r = None
        # For each distance
        for distance in range(row['end'] - row['start'] + 1):
            # Skip the case of distance equals to zero
            if distance == 0: 
                continue
            # Calculate the parameters for the negative binomial distribution
            
            mean = distance_means[row['chr']].get(distance)
            var = distance_vars[row['chr']].get(distance)
            if var != None and mean != None and var > mean and var != 0:
                p = mean/var
                r = mean**2/(var-mean)
            else:
                p, r = prev_p, prev_r
            prev_p, prev_r = p, r

            # For each pair of bins at this distance
            for i in range(row['start'], row['end'] - distance + 1):
                j = i + distance                
                # Generate a value from the negative binomial distribution
                new_data.loc[i, j] = np.random.negative_binomial(r, p)
                new_data.loc[j, i] = new_data.loc[i, j]


    # For the interchromosomal regions
    p = interchr_mean / interchr_var
    r = interchr_mean**2 / (interchr_var - interchr_mean)

    # Get the upper triangular indices excluding the diagonal
    i_upper, j_upper = np.triu_indices(new_data.shape[0], 1)
    
    # Find the indices of the 0 values in the upper triangle
    zero_indices = np.where(new_data.values[i_upper, j_upper] == 0)
    
    # Compute the random values to assign
    values = nbinom.rvs(n=r, p=p, size=zero_indices[0].size)
    
    # Assign the random values to the 0 cells in the upper triangle
    new_data.values[i_upper[zero_indices], j_upper[zero_indices]] = values
    
    # To keep the matrix symmetric, assign the same values to the lower triangle
    new_data.values[j_upper[zero_indices], i_upper[zero_indices]] = values


    # Fill the diagonal with zeros
    np.fill_diagonal(new_data.values, 0)

    return new_data

