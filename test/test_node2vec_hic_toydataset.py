import unittest
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from node2vec_hic_toydataset import data_to_embedding

class TestDataToEmbedding(unittest.TestCase):
    def setUp(self):
        """
        This method is called before each test. Here we setup variables that will be used in the test methods.
        """
        self.data = pd.DataFrame(np.random.randint(0, 2, size=(5, 5)))

    def test_data_to_embedding(self):
        """
        Test the function 'data_to_embedding'.

        This test case checks the correct transformation of adjacency data to a Node2Vec embedding and subsequent PCA dimension reduction. 
        We verify the function by asserting the shape and type of the returned numpy array.
        """
        n2v_dimensions = 4
        dimensions = 2
        walk_length = 10
        num_walks = 50
        workers = 1
        p = 1
        q = 1

        embedding = data_to_embedding(self.data, n2v_dimensions, dimensions, walk_length, num_walks, workers, p, q)
        
        # Assert that the returned object is a numpy array
        self.assertIsInstance(embedding, np.ndarray)
        
        # Assert that the shape of the embedding is as expected
        self.assertEqual(embedding.shape, (self.data.shape[0], dimensions))

if __name__ == '__main__':
    unittest.main()
