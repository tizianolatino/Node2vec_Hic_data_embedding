#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:33:41 2023

@author: tizianolatino
"""

import pandas as pd
import os

from negative_binomial_model import *
from Hic_data_prepro import *
from utils import *

def main(data, metadata):

   # Drop the specified chromosome and adjust the start and end points of the remaining chromosomes
   metadata, data = drop_chr(metadata, data, "'chrY'")

   # Remove isolated bins and adjust the start and end points of the chromosomes
   metadata, data = remove_isolated_bins(metadata, data)
   
   #reduce dimension
   data, metadata = reduce_dimension(metadata, data, 0.5)

   # Calculate the mean and variance for each distance within each chromosome, 
   # and for interchromosomal interactions
   intrachr_means, intrachr_vars = intrachr_contacts_mean_var(metadata, data)
   interchr_mean, interchr_var = interchr_contacts_mean_var(metadata, data)

   # Generate a new dataframe with values from a negative binomial distribution
   #neg_bin_data = generate_data_from_neg_binomial(metadata, data)

   # Save the new data
   '''
   if not os.path.exists("Data"):
        os.makedirs("Data")
   neg_bin_data.to_csv("Data/neg_bin_data.csv")'''
    
   return data, metadata

if __name__ == "__main__":
    
    # Load the metadata and data
    data = pd.read_csv('Data/raw_GM12878_1Mb.csv', header=None)
    metadata = pd.read_csv('Data/metadata.csv')
    
    neg_bin_data, neg_bin_metadata = main(data, metadata)