import unittest
import pandas as pd
import numpy as np
from scipy.stats import nbinom
import sys
sys.path.append('..')
from negative_binomial_model import intrachr_contacts_mean_var, interchr_contacts_mean_var, generate_data_from_neg_binomial
from utils_test import assertAlmostEqualRelative, validate_neg_binomial_distribution

class TestHiCDataFunctions(unittest.TestCase):
    """
    TestHiCDataFunctions is a test case class which inherits from unittest.TestCase. This class
    tests various functions related to manipulating Hi-C contact data.
    """
    
    def setUp(self):
        """
        setUp is a method that unittest.TestCase calls for you just before each test method is run.
        Here, it sets up a random dataset and the associated metadata.
        """
        self.metadata = pd.DataFrame({'chr': ['chr1', 'chr2','chr3'], 'start': [0, 100, 200], 
                                      'end': [99, 199, 299]})
        np.random.seed(42)
        self.data = pd.DataFrame(np.random.randint(0, high=100, size=(300, 300)))
        
    def test_intrachr_contacts_mean_var(self):
        """
        Test function for 'intrachr_contacts_mean_var'.

        This test asserts that the output dictionaries have the expected number of chromosomes (keys in the outer dictionary),
        the expected number of distances (keys in the inner dictionary), and all the mean and variance values are non-negative.
        """
        mean, var = intrachr_contacts_mean_var(self.metadata, self.data)

        # Assert that the returned dictionaries have three keys corresponding to the three chromosomes
        self.assertEqual(set(mean.keys()), {'chr1', 'chr2', 'chr3'})
        self.assertEqual(set(var.keys()), {'chr1', 'chr2', 'chr3'})

        # Assert that each chromosome dictionary has 100 keys corresponding to the 100 distances
        self.assertEqual(set(mean['chr1'].keys()), set(range(100)))
        self.assertEqual(set(var['chr1'].keys()), set(range(100)))

        # Assert that all the mean and variance values are non-negative
        for chr in mean:
            self.assertTrue(all(m >= 0 for m in mean[chr].values()))
            self.assertTrue(all(v >= 0 for v in var[chr].values()))

    def test_interchr_contacts_mean_var(self):
        """
        test_interchr_contacts_mean_var tests the interchr_contacts_mean_var function by checking
        that the returned mean and variance are the correct type and have non-negative values.
        """
        mean, var = interchr_contacts_mean_var(self.metadata, self.data)
        self.assertIsInstance(mean, float)
        self.assertIsInstance(var, float)
        # check that the mean and variance are non-negative
        self.assertTrue(mean >= 0)
        self.assertTrue(var >= 0)

    def test_generate_data_from_neg_binomial(self):
        """
        test_generate_data_from_neg_binomial tests the generate_data_from_neg_binomial function by checking
        that the generated data has the correct shape, contains zeros on the diagonal, and matches the
        negative binomial distribution.
        """
        mean, var = intrachr_contacts_mean_var(self.metadata, self.data)
        interchr_mean, interchr_var = interchr_contacts_mean_var(self.metadata, self.data)
        
        neg_bin_data = generate_data_from_neg_binomial(self.metadata, self.data, mean, var, interchr_mean, interchr_var)
        self.assertEqual(self.data.shape, neg_bin_data.shape)
        self.assertTrue((np.diag(neg_bin_data.values) == 0).all())
        try:
            result = validate_neg_binomial_distribution(self.metadata, self.data, neg_bin_data)
            self.assertTrue(result)
        except AssertionError as error:
            self.fail(str(error))

    def test_empty_data(self):
        """
        Test function when input data is empty.
        """
        metadata = pd.DataFrame({'chr': [], 'start': [], 'end': []})
        data = pd.DataFrame()
        
        mean, var = intrachr_contacts_mean_var(metadata, data)
        self.assertEqual(mean, {})
        self.assertEqual(var, {})

        mean, var = interchr_contacts_mean_var(metadata, data)
        self.assertTrue(np.isnan(mean))
        self.assertTrue(np.isnan(var))

    def test_all_zeros_data(self):
        """
        Test function when input data contains all zeros.
        """
        metadata = self.metadata
        data = pd.DataFrame(np.zeros((300, 300)))
        
        distance_means, distance_vars = intrachr_contacts_mean_var(metadata, data)
        for chr in distance_means:
            self.assertTrue(all(m == 0 for m in distance_means[chr].values()))
            self.assertTrue(all(v == 0 for v in distance_vars[chr].values()))
        
        interchr_mean, interchr_var = interchr_contacts_mean_var(metadata, data)
        self.assertEqual(interchr_mean, 0.0)
        self.assertEqual(interchr_var, 0.0)


        with self.assertRaises(ValueError):
            neg_bin_data = generate_data_from_neg_binomial(metadata, data, distance_means, distance_vars, interchr_mean, interchr_var)

    def test_data_with_na_values(self):
        """
        Test function when input data contains NA values.
        """
        metadata = self.metadata
        data = self.data.copy()
        data.iloc[0, 0] = np.nan
        
        with self.assertRaises(ValueError):
            intrachr_contacts_mean_var(metadata, data)

        with self.assertRaises(ValueError):
            interchr_contacts_mean_var(metadata, data)

if __name__ == '__main__':
    unittest.main()
