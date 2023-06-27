#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:33:41 2023

@author: tizianolatino
"""

import pandas as pd
import os

from negative_binomial_model import generate_data_from_neg_binomial
from Hic_data_prepro import drop_chr, remove_isolated_bins
from utils import reduce_dimension
from utils_data_metadata import adjust_indices_to_zero
from draw import plot_embedding, plot_data
from translocated_data import generate_node_labels
from node2vec_hic_toydataset import data_to_embedding


def main(data, metadata):
    
   metadata, data = adjust_indices_to_zero(metadata, data)

   # Drop the specified chromosome and adjust the start and end points of the remaining chromosomes
   metadata, data = drop_chr(metadata, data, "'chrY'")

   # Remove isolated bins and adjust the start and end points of the chromosomes
   metadata, data = remove_isolated_bins(metadata, data)
   
   #reduce dimension
   data, metadata = reduce_dimension(metadata, data, 0.9)

   # Generate a new dataframe with values from a negative binomial distribution
   neg_bin_data = generate_data_from_neg_binomial(metadata, data)
   
   
   node_labels = generate_node_labels(metadata, "'chr1'", 5, 10, "'chr6'", 105,110)
   
   # Plot the Adajacency Matrix
   plot_data(neg_bin_data, metadata, [0,10,100,1000,5000,10000])
   
   
   embeddings = data_to_embedding(data)
   
   plot_embedding(embeddings, node_labels)
   
   return neg_bin_data, metadata

if __name__ == "__main__":
    
    # Load the metadata and data
    data = pd.read_csv('Data/raw_GM12878_1Mb.csv', header=None)
    metadata = pd.read_csv('Data/metadata.csv')
    

    #neg_bin_data, neg_bin_metadata = main(data, metadata)

    

    