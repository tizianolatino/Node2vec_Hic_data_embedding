#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 17:50:37 2023

@author: tizianolatino
"""

import unittest
import pandas as pd
import numpy as np
from Hic_data_prepro import *

class TestChromosomeFunctions(unittest.TestCase):

    def setUp(self):
        self.metadata = pd.DataFrame({
            'chr': ['chr1', 'chr2', 'chrY', 'chr3'],
            'start': [0, 100, 200, 300],
            'end': [99, 199, 299, 399]
        })
        # Create a 400x400 adjacency matrix with random values and some rows and columns of zeros
        self.data = pd.DataFrame(np.random.rand(400, 400), index=range(400), columns=range(400))
        self.data.iloc[50:60, :] = 0  # 10 bins in 'chr1' are isolated
        self.data.iloc[:, 50:60] = 0
        self.data.iloc[250:260, :] = 0  # 10 bins in 'chrY' are isolated
        self.data.iloc[:, 250:260] = 0

    def test_drop_chr(self):
        updated_metadata, updated_data = drop_chr(self.metadata, self.data, 'chrY')

        # Check that 'chrY' is removed
        self.assertTrue('chrY' not in updated_metadata['chr'].values)

        # Check that the start and end points for 'chr3' are adjusted
        chr3_row = updated_metadata[updated_metadata['chr'] == 'chr3']
        self.assertEqual(chr3_row['start'].values[0], 200)
        self.assertEqual(chr3_row['end'].values[0], 299)

        # Check that the bins for 'chrY' are removed from the adjacency matrix
        self.assertEqual(updated_data.shape, (300, 300))

    def test_remove_isolated_bins(self):
        updated_metadata, updated_data = remove_isolated_bins(self.metadata, self.data)

        # Check that the isolated bins are removed from the adjacency matrix
        self.assertEqual(updated_data.shape, (380, 380))

        # Check that the 'end' point for 'chr1' is adjusted
        chr1_row = updated_metadata[updated_metadata['chr'] == 'chr1']
        self.assertEqual(chr1_row['end'].values[0], 89)

        # Check that the start and end points for the remaining chromosomes are adjusted
        chr2_row = updated_metadata[updated_metadata['chr'] == 'chr2']
        self.assertEqual(chr2_row['start'].values[0], 90)
        self.assertEqual(chr2_row['end'].values[0], 189)
        chrY_row = updated_metadata[updated_metadata['chr'] == 'chrY']
        self.assertEqual(chrY_row['start'].values[0], 190)
        self.assertEqual(chrY_row['end'].values[0], 279)
        chr3_row = updated_metadata[updated_metadata['chr'] == 'chr3']
        self.assertEqual(chr3_row['start'].values[0], 280)
        self.assertEqual(chr3_row['end'].values[0], 379)

if __name__ == '__main__':
    unittest.main()
