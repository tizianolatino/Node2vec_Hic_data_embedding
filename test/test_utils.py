import pandas as pd
import numpy as np
import unittest
import sys
sys.path.append('..')
from utils import reduce_dimension

class TestReduceDimension(unittest.TestCase):

    def setUp(self):
        """
        Set up a common test environment for all test cases.
        """
        self.metadata = pd.DataFrame({
            'chr': ['chr1', 'chr2', 'chr3'],
            'start': [0, 100, 200],
            'end': [99, 199, 299]
        })
        self.data = pd.DataFrame(np.random.randint(0, 100, (300, 300)))

    def test_reduce_dimension(self):
        """
        Description: This test checks whether the function correctly reduces the dimension of the data
        by a given percent for each chromosome, which involves reducing the number of bins, and adjusting the
        start and end indices in the metadata accordingly. The assertions are checking:
        1. The shape of the new_data DataFrame is correctly reduced.
        2. The start and end indices in the new_metadata are correctly adjusted.
        """
        percent = 0.1  # Reduce each chromosome by 10%
        new_data, new_metadata = reduce_dimension(self.metadata, self.data, percent)

        # Check that the dimensions of the new_data DataFrame are correct.
        expected_data_shape = (270, 270)  # Each chromosome was reduced by 10 bins.
        self.assertEqual(new_data.shape, expected_data_shape)

        # Check that the start and end indices in new_metadata are correct.
        expected_metadata = pd.DataFrame({
            'chr': ['chr1', 'chr2', 'chr3'],
            'start': [0, 90, 180],
            'end': [89, 179, 269]
        })
        pd.testing.assert_frame_equal(new_metadata, expected_metadata)

    def test_empty_metadata(self):
        """
        Description: This test checks whether the function correctly handles an empty metadata DataFrame.
        The output data and metadata should also be empty.
        """
        empty_metadata = pd.DataFrame()
        empty_data = pd.DataFrame()

        new_data, new_metadata = reduce_dimension(empty_metadata, empty_data, 0.1)

        self.assertTrue(new_data.empty)
        self.assertTrue(new_metadata.empty)

    def test_zero_percent(self):
        """
        Description: This test checks whether the function correctly handles a zero reduction percent.
        The output data and metadata should be identical to the input data and metadata.
        """
        percent = 0.0
        new_data, new_metadata = reduce_dimension(self.metadata, self.data, percent)

        pd.testing.assert_frame_equal(new_data, self.data)
        pd.testing.assert_frame_equal(new_metadata, self.metadata)

    def test_full_reduction(self):
        """
        Description: This test checks whether the function correctly handles a full reduction percent.
        The output data and metadata should be empty.
        """
        percent = 1.0
        new_data, new_metadata = reduce_dimension(self.metadata, self.data, percent)
    
        self.assertTrue(new_data.empty)
        self.assertTrue(new_metadata.empty)

    def test_over_full_reduction(self):
        """
        Description: This test checks whether the function correctly handles a reduction percent greater than 1.
        It should raise a ValueError.
        """
        percent = 1.1
        with self.assertRaises(ValueError):
            new_data, new_metadata = reduce_dimension(self.metadata, self.data, percent)
            print(new_data,new_metadata)

    def test_negative_reduction(self):
        """
        Description: This test checks whether the function correctly handles a negative reduction percent.
        It should raise a ValueError.
        """
        percent = -0.1
        with self.assertRaises(ValueError):
            new_data, new_metadata = reduce_dimension(self.metadata, self.data, percent)

if __name__ == "__main__":
    unittest.main()
