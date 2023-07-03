#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:33:41 2023

@author: tizianolatino
"""

import pandas as pd
import os
import argparse

from negative_binomial_model import generate_data_from_neg_binomial, intrachr_contacts_mean_var, interchr_contacts_mean_var
from Hic_data_prepro import drop_chr, remove_isolated_bins
from utils import reduce_dimension
from utils_data_metadata import adjust_indices_to_zero
from draw import plot_embedding, plot_data
from translocated_data import apply_RBT_modifications, generate_node_labels
from node2vec_hic_toydataset import data_to_embedding


def main(data, metadata):
    
   metadata_ori, data_ori = adjust_indices_to_zero(metadata, data)

   # Drop the specified chromosome and adjust the start and end points of the remaining chromosomes
   metadata_ori, data_ori = drop_chr(metadata_ori, data_ori, "'chrY'")

   # Remove isolated bins and adjust the start and end points of the chromosomes
   metadata_ori, data_ori = remove_isolated_bins(metadata_ori, data_ori)
   
   # Calculate the mean and variance for each distance within each chromosome, 
   # and for interchromosomal interactions
   intrachr_means, intrachr_vars = intrachr_contacts_mean_var(metadata_ori, data_ori)
   interchr_mean, interchr_var = interchr_contacts_mean_var(metadata_ori, data_ori)


   # Generate a new dataframe with values from a negative binomial distribution
   neg_bin_data = generate_data_from_neg_binomial(metadata_ori, data_ori, intrachr_means, intrachr_vars,
                                                  interchr_mean, interchr_var )
   
   # Reduce dimension
   data_red, metadata_red = reduce_dimension(metadata_ori, neg_bin_data, 0.9)
   
   #Introduce rbt translocation on the data
   data_toy_trans = apply_RBT_modifications(data_red, metadata_red, "'chr1'", 5, 10, "'chr6'", 105, 110,
                                            intrachr_means, intrachr_vars,interchr_mean, interchr_var)                                                                
   
   # Generates labels for each bin in the data
   node_labels = generate_node_labels(metadata_red, "'chr1'", 5, 10, "'chr6'", 105, 110)
   
   # Plot the Adajacency Matrix
   plot_data(data_toy_trans, metadata_red, [0, 10, 100, 1000, 5000, 10000])
   
   # Node Embeddings and PCA data reduction
   embeddings = data_to_embedding( data_toy_trans)
   
   # Plots a 2D  of the embeddings
   plot_embedding(embeddings, node_labels)
   
   return neg_bin_data, metadata, data_toy_trans, embeddings

if __name__ == "__main__":
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Read two CSV files.')
    
    # Add the arguments
    parser.add_argument('csv1', type=str, help='The path to the first CSV file')
    parser.add_argument('csv2', type=str, help='The path to the second CSV file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Read the CSV files using pandas
    data = pd.read_csv(args.csv1,  header=None)
    metadata = pd.read_csv(args.csv2)
    
    # Print the data frames
    print("Shape of the first CSV file:")
    print(data.shape)
    print("\nShape of the second CSV file:")
    print(metadata.shape)

     

    neg_bin_data, neg_bin_metadata, data_toy_trans, embeddings = main(data, metadata)

    

    