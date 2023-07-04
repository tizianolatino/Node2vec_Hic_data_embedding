import unittest
import sys
sys.path.append('..')
from utils_statistics import calculate_parameters

class TestCalculateParameters(unittest.TestCase):

    def setUp(self):
        """
        Set up testing parameters for the test cases.
        """
        self.prev_p = 0.5
        self.prev_r = 0.5

    def test_calculate_parameters_with_valid_inputs(self):
        """
        Test calculate_parameters function with valid inputs where variance is 
        greater than mean and variance is not zero. It should return new p and r values.
        """
        result = calculate_parameters(10, 20, self.prev_p, self.prev_r)
        self.assertAlmostEqual(result[0], 0.5)
        self.assertAlmostEqual(result[1], 10)

    def test_calculate_parameters_with_zero_variance(self):
        """
        Test calculate_parameters function with zero variance. It should return 
        previous p and r values as the new p and r values.
        """
        result = calculate_parameters(10, 0, self.prev_p, self.prev_r)
        self.assertAlmostEqual(result[0], self.prev_p)
        self.assertAlmostEqual(result[1], self.prev_r)

    def test_calculate_parameters_with_mean_greater_than_variance(self):
        """
        Test calculate_parameters function with mean greater than variance. 
        It should return previous p and r values as the new p and r values.
        """
        result = calculate_parameters(20, 10, self.prev_p, self.prev_r)
        self.assertAlmostEqual(result[0], self.prev_p)
        self.assertAlmostEqual(result[1], self.prev_r)

    def test_calculate_parameters_with_null_inputs(self):
        """
        Test calculate_parameters function with null inputs for mean and variance. 
        It should return previous p and r values as the new p and r values.
        """
        result = calculate_parameters(None, None, self.prev_p, self.prev_r)
        self.assertAlmostEqual(result[0], self.prev_p)
        self.assertAlmostEqual(result[1], self.prev_r)

if __name__ == '__main__':
    unittest.main()


