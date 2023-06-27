import pandas as pd
import numpy as np
import unittest
import sys
sys.path.append('..')
from utils import reduce_dimension

class TestReduceDimension(unittest.TestCase):

    def test_reduce_dimension(self):
        """
        Description: This test checks whether the function correctly reduces the dimension of the data
        by a given percent for each chromosome, which involves reducing the number of bins, and adjusting the
        start and end indices in the metadata accordingly. The assertions are checking:
        1. The shape of the new_data DataFrame is correctly reduced.
        2. The start and end indices in the new_metadata are correctly adjusted.
        """

        # Step 1: Create a metadata DataFrame.
        metadata = pd.DataFrame({
            'chr': ['chr1', 'chr2', 'chr3'],
            'start': [0, 100, 200],
            'end': [99, 199, 299]
        })

        # Step 2: Create a DataFrame to represent the adjacency matrix (data).
        data = pd.DataFrame(np.random.randint(0, 100, (300, 300)))

        # Step 3: Apply the reduce_dimension function.
        percent = 0.1  # Reduce each chromosome by 10%
        new_data, new_metadata = reduce_dimension(metadata, data, percent)

        # Step 4: Check that the dimensions of the new_data DataFrame are correct.
        expected_data_shape = (270, 270)  # Each chromosome was reduced by 10 bins.
        self.assertEqual(new_data.shape, expected_data_shape)

        # Step 5: Check that the start and end indices in new_metadata are correct.
        expected_metadata = pd.DataFrame({
            'chr': ['chr1', 'chr2', 'chr3'],
            'start': [0, 90, 180],
            'end': [89, 179, 269]
        })
        pd.testing.assert_frame_equal(new_metadata, expected_metadata)

if __name__ == "__main__":
    unittest.main()
