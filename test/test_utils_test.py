import unittest
import pandas as pd
import numpy as np
from  utils_test import assertAlmostEqualRelative, validate_neg_binomial_distribution

class TestYourModule(unittest.TestCase):
    """
    Unit Test class for the functions 'assertAlmostEqualRelative' and 'validate_neg_binomial_distribution' from 'your_module' module.
    """
    def setUp(self):
        """
        Set up function to create the test data. This method initializes DataFrame 'metadata', 'original_data' and 'neg_bin_data' for testing.
        """
        self.metadata = pd.DataFrame({'chr': ['chr1'], 'start': [0], 'end': [3]})
        self.original_data = pd.DataFrame(np.array([[1, 2, 3, 4], [2, 5, 6, 7], [3, 6, 8, 10], [4, 7, 10, 12]]))
        self.neg_bin_data = pd.DataFrame(np.array([[1, 2, 3, 4], [2, 5, 6, 7], [3, 6, 8, 10], [4, 7, 10, 12]]))

    def test_assertAlmostEqualRelative(self):
        """
        Test function for 'assertAlmostEqualRelative'.
    
        This test asserts that the function is able to correctly determine if two numbers are approximately equal 
        relative to the magnitude of the first number.
        """
        # This should not raise an exception
        assertAlmostEqualRelative(self, 100, 101, rel_tol=0.02)
        
        # This should raise an exception
        with self.assertRaises(AssertionError):
            assertAlmostEqualRelative(self, 100, 105, rel_tol=0.02)


    def test_validate_neg_binomial_distribution(self):
        """
        Test function for 'validate_neg_binomial_distribution'.

        This test asserts that the function can correctly validate that the original data matches the distribution 
        of the generated data.
        """
        self.assertTrue(validate_neg_binomial_distribution(self.metadata, self.original_data, self.neg_bin_data))
        
        # Modify the neg_bin_data to make it not match the original_data
        self.neg_bin_data.loc[1, 1] = 1000
        self.assertRaises(AssertionError, validate_neg_binomial_distribution, self.metadata, self.original_data, self.neg_bin_data)

if __name__ == '__main__':
    unittest.main()
