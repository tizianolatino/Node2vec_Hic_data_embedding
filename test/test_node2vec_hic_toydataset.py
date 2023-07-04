import unittest
import pandas as pd
import numpy as np
import sys
import networkx as nx
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

    def test_negative_dimensions(self):
        """
        Test that an error is raised when `n2v_dimensions` is less than `dimensions`.
        """
        with self.assertRaises(ValueError):
            data_to_embedding(self.data, 2, 4, 10, 50, 1, 1, 1)
            
    def test_zero_dimensions(self):
        """
        Test that an error is raised when `dimensions` is zero.
        """
        with self.assertRaises(ValueError):
            data_to_embedding(self.data, 4, 0, 10, 50, 1, 1, 1)

    def test_negative_walk_length(self):
        """
        Test that an error is raised when `walk_length` is negative.
        """
        with self.assertRaises(ValueError):
            data_to_embedding(self.data, 4, 2, -1, 50, 1, 1, 1)

    def test_zero_num_walks(self):
        """
        Test that an error is raised when `num_walks` is zero.
        """
        with self.assertRaises(RuntimeError) as context:
            data_to_embedding(self.data, 4, 2, 10, 0, 1, 1, 1)
        self.assertEqual(str(context.exception), "you must first build vocabulary before training the model")

    def test_negative_p(self):
        """
        Test that an error is raised when `p` is negative.
        """
        with self.assertRaises(ValueError):
            data_to_embedding(self.data, 4, 2, 10, 50, 1, -1, 1)

    def test_non_square_data(self):
        """
        Test that an error is raised when `data` is not a square DataFrame.
        """
        with self.assertRaises(nx.NetworkXError) as context:
            data_to_embedding(pd.DataFrame(np.random.randint(0, 2, size=(5, 4))), 4, 2, 10, 50, 1, 1, 1)
        self.assertIn("Adjacency matrix not square", str(context.exception))

    def test_empty_data(self):
        """
        Test that an error is raised when `data` is an empty DataFrame.
        """
        with self.assertRaises(RuntimeError):
            data_to_embedding(pd.DataFrame(), 4, 2, 10, 50, 1, 1, 1)

    
    def test_negative_n2v_dimensions(self):
        """
        Test that an error is raised when `n2v_dimensions` is negative.
        """
        with self.assertRaises(ValueError):
            data_to_embedding(self.data, -4, 2, 10, 50, 1, 1, 1)

    def test_negative_num_walks(self):
        """
        Test that an error is raised when `num_walks` is negative.
        """
        with self.assertRaises(RuntimeError):
            data_to_embedding(self.data, 4, 2, 10, -50, 1, 1, 1)

    def test_negative_p_and_q(self):
        """
        Test that an error is raised when `p` and `q` are negative.
        """
        with self.assertRaises(ValueError):
            data_to_embedding(self.data, 4, 2, 10, 50, 1, -1, -1)

    def test_zero_p_and_q(self):
        """
        Test that an error is raised when `p` and `q` are zero.
        """
        with self.assertRaises(ZeroDivisionError):
            data_to_embedding(self.data, 4, 2, 10, 50, 1, 0, 0)

    def test_data_to_embedding_(self):
        """
        Test the function 'data_to_embedding'.

        This test case checks the correct transformation of adjacency data to a Node2Vec embedding and subsequent PCA dimension reduction. 
        We verify the function by asserting the shape and type of the returned numpy array.
        """
        embedding = data_to_embedding(self.data, 4, 2, 10, 50, 1, 1, 1)
        
        # Assert that the returned object is a numpy array
        self.assertIsInstance(embedding, np.ndarray)
        
        # Assert that the shape of the embedding is as expected
        self.assertEqual(embedding.shape, (self.data.shape[0], 2))

if __name__ == '__main__':
    unittest.main()


