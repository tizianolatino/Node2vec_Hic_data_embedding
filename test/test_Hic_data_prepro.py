import unittest
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from Hic_data_prepro import drop_chr, remove_isolated_bins, remove_isolated_bins

class TestChromosomeFunctions(unittest.TestCase):
    """
    TestChromosomeFunctions is a test case class which inherits from unittest.TestCase.
    This class is designed to test various functions related to manipulating chromosomal data.
    """


    def setUp(self):
        """
        setUp is a special method that unittest.TestCase calls for you just before each test method is run. 
        It includes code to create and configure the test environment so that everything is ready when the test methods are run.
        """
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
        """
        test_drop_chr tests the function drop_chr by checking whether 'chrY' has been removed and 
        the start and end points for 'chr3' are adjusted in the metadata. It also checks if 'chrY' bins 
        are removed from the adjacency matrix in the data.
        """
        
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
        """
        test_remove_isolated_bins tests the function remove_isolated_bins by verifying if the 
        isolated bins are removed from the adjacency matrix, and the 'end' point for 'chr1' and 
        the start and end points for the remaining chromosomes are adjusted in the metadata.
        """
        
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
        
    def test_drop_chr_nonexistent(self):
        """
        test_drop_chr_nonexistent tests the function drop_chr by attempting to drop a chromosome that does not exist in the metadata.
        It expects to receive a ValueError.
        """
        
        with self.assertRaises(ValueError):
            drop_chr(self.metadata, self.data, 'chrX')  # 'chrX' does not exist in the metadata

    def test_remove_isolated_bins_none(self):
        """
        test_remove_isolated_bins_none tests the function remove_isolated_bins by passing data with no isolated bins.
        It expects the output data and metadata to be the same as the input.
        """

        # Remove the isolated bins in the setup data
        no_isolated_data = self.data.copy(deep=True)
        no_isolated_data.iloc[50:60, :] = np.random.rand(10, 400)
        no_isolated_data.iloc[:, 50:60] = np.random.rand(400, 10)
        no_isolated_data.iloc[250:260, :] = np.random.rand(10, 400)
        no_isolated_data.iloc[:, 250:260] = np.random.rand(400, 10)

        updated_metadata, updated_data = remove_isolated_bins(self.metadata, no_isolated_data)

        # The updated data should be the same as no_isolated_data
        pd.testing.assert_frame_equal(updated_data, no_isolated_data)

        # The updated metadata should be the same as the setup metadata
        pd.testing.assert_frame_equal(updated_metadata, self.metadata)
   

if __name__ == '__main__':
    unittest.main()
