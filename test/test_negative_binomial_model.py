#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:03:20 2023

@author: tizianolatino
"""

import unittest
import pandas as pd
import numpy as np
from scipy.stats import nbinom
from negative_binomial_model import *

class TestHiCDataFunctions(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame({'chr': ['chr1', 'chr2','chr3'], 'start': [0, 100, 200], 
                                      'end': [99, 199, 299]})
        self.data = pd.DataFrame(np.random.randint(0, high=100, size=(300, 300)))
    def test_intrachr_contacts_mean_var(self):
        mean, var = intrachr_contacts_mean_var(self.metadata, self.data)
        self.assertEqual(mean.shape, var.shape)
        self.assertEqual(mean.index.tolist(), ['chr1', 'chr2', 'chr3'])
        # check that the mean and variance dataframes have non-negative values
        self.assertTrue((mean >= 0).all().all())
        self.assertTrue((var >= 0).all().all())

    def test_interchr_contacts_mean_var(self):
        mean, var = interchr_contacts_mean_var(self.metadata, self.data)
        self.assertIsInstance(mean, float)
        self.assertIsInstance(var, float)
        # check that the mean and variance are non-negative
        self.assertTrue(mean >= 0)
        self.assertTrue(var >= 0)
        
    def assertAlmostEqualRelative(self, a, b, rel_tol=1e-2):
        """
        Checks if 'a' and 'b' are approximately equal relative to the magnitude of 'a'.
    
        Args:
        a, b: The two numbers to compare.
        rel_tol: The relative tolerance -- the maximum allowed difference between 'a' and 'b',
                 relative to the magnitude of 'a'.
        """
        diff = abs(a - b)
        self.assertLessEqual(diff, abs(rel_tol * a))

    def test_generate_data_from_neg_binomial(self):
        neg_bin_data = generate_data_from_neg_binomial(self.metadata, self.data)
        self.assertEqual(self.data.shape, neg_bin_data.shape)
        self.assertTrue((np.diag(neg_bin_data.values) == 0).all())
        # Additional test: check that the generated data matches the negative binomial distribution
        for _, row in self.metadata.iterrows():
            for distance in range(row['end'] - row['start'] + 1):
                if distance == 0 or distance >10: continue
                # Select the data for this distance
                original_values = self.data.loc[row['start']:row['end'], row['start']:row['end']].iloc[::distance].values.flatten()
                generated_values = neg_bin_data.loc[row['start']:row['end'], row['start']:row['end']].iloc[::distance].values.flatten()
    
                # Calculate the mean and variance for the original and generated values
                mean_original = original_values.mean()
                var_original = original_values.var()
                mean_generated = generated_values.mean()
                var_generated = generated_values.var()
                
                # Check that the parameters are valid for a negative binomial distribution
                if mean_original > 0 and var_original > mean_original:
                    p_original = mean_original/var_original
                    r_original = mean_original**2/(var_original-mean_original)
                    mean_nbinomial = nbinom.mean(r_original, p_original)
                    var_nbinomial = nbinom.var(r_original, p_original)
                    self.assertAlmostEqualRelative(mean_nbinomial, mean_generated, rel_tol=0.1)
                    self.assertAlmostEqualRelative(var_nbinomial, var_generated, rel_tol=0.1)
                else:
                    print(f"Invalid parameters: mean = {mean_original}, variance = {var_original}")
                    self.skipTest("The original data does not have valid parameters for a negative binomial distribution")


if __name__ == '__main__':
    unittest.main()