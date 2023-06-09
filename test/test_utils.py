#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:54:35 2023

@author: tizianolatino
"""

import unittest
import pandas as pd
import numpy as np
from utils import *

class TestReduceDimension(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame({'chr': ['chr1', 'chr2', 'chr3', 'chr4'],
                                      'start': [0, 100, 200, 300],
                                      'end': [99, 199, 299, 399]})
        self.data = pd.DataFrame(index=range(400), columns=range(400))

    def test_reduce_dimension(self):
        # Test that the dimensions are reduced correctly
        reduced_data, reduced_metadata = reduce_dimension(self.metadata, self.data, 0.1)

        # Assert that the dimensions of the data are reduced
        self.assertEqual(reduced_data.shape, (360, 360))

        # Assert that the dimensions of the metadata are the same
        self.assertEqual(reduced_metadata.shape, self.metadata.shape)

        for i, row in reduced_metadata.iterrows():
            # Check the new 'start' and 'end' values
            self.assertEqual(row['start'], i * 90)
            self.assertEqual(row['end'], (i + 1) * 90 - 1)
            
            
            # Check if the number of rows/columns in the reduced data equals to the last 'end' index + 1
            self.assertEqual(reduced_data.shape[0], reduced_metadata.iloc[-1]['end'] + 1)
            self.assertEqual(reduced_data.shape[1], reduced_metadata.iloc[-1]['end'] + 1)



if __name__ == '__main__':
    unittest.main()
