import unittest
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from translocated_data import generate_node_labels,get_translocated_bins, modify_interchr_contacts, modify_intrachr_contacts, apply_RBT_modifications
from negative_binomial_model import intrachr_contacts_mean_var, interchr_contacts_mean_var

class TestTranslocation(unittest.TestCase):
    def setUp(self):
        """
        This method is called before each test. Here we setup variables that will be used in the test methods.
        """
        
        self.metadata = pd.DataFrame({
            'chr': ['chr1', 'chr2'],
            'start': [0, 100],
            'end': [99, 199]
        })
        
        np.random.seed(42)
        self.data = pd.DataFrame(np.random.randint(0, 10, (200, 200))) 
        self.intrachr_means, self.intrachr_vars = intrachr_contacts_mean_var(self.metadata, self.data)
        self.interchr_mean, self.interchr_var = interchr_contacts_mean_var(self.metadata, self.data)

    def test_get_translocated_bins(self):
        """
        Test get_translocated_bins function to ensure it returns correct bins after translocation.
        """
        chr1_bins, chr2_bins = get_translocated_bins(self.metadata, 'chr1', 10, 20, 'chr2', 130, 140)
        self.assertEqual(len(chr1_bins), 100)
        self.assertEqual(len(chr2_bins), 100)

    def test_modify_interchr_contacts(self):
        """
        Test modify_interchr_contacts function to ensure it correctly modifies the contact data.
        """
        modified_data = modify_interchr_contacts(self.data, self.metadata, 'chr1', 10, 20, 'chr2', 130, 140, self.intrachr_means, self.intrachr_vars)
        self.assertFalse(modified_data.equals(self.data))

    def test_modify_intrachr_contacts(self):
        """
        Test modify_intrachr_contacts function to ensure it correctly modifies the contact data.
        """
        modified_data = modify_intrachr_contacts(self.metadata, self.data, 'chr1', 10, 20, 'chr2', 130, 140, self.interchr_mean, self.interchr_var)
        self.assertFalse(modified_data.equals(self.data))

    def test_apply_RBT_modifications(self):
        """
        Test apply_RBT_modifications function to ensure it correctly modifies the contact data.
        """
        modified_data = apply_RBT_modifications(self.data, self.metadata, 'chr1', 10, 20, 'chr2', 130, 140, self.intrachr_means, self.intrachr_vars, self.interchr_mean, self.interchr_var)
        self.assertFalse(modified_data.equals(self.data))


class TestGenerateNodeLabels(unittest.TestCase):
    def setUp(self):
        """
        This method is called before each test. Here we setup variables that will be used in the test methods.
        """
        self.metadata = pd.DataFrame({'chr': ['chr1', 'chr2', 'chr3', 'chr4', 'chr5'], 
                                      'start': [0, 100, 200, 300, 400], 
                                      'end': [99, 199, 299, 399, 499]})
        self.chr1, self.chr2 = 'chr2', 'chr4'
        self.start1, self.start2 = 120, 340
        self.end1, self.end2 = 150, 370

    def test_generate_node_labels(self):
        """
        Test the function 'generate_node_labels'.

        This test case checks the correct assignment of labels for each bin.
        In the presence of a Rearrangement Breakpoint (RBT), this function should assign a negative label for the bins within the RBT region.
        We verify if the function generates the correct labels for each bin based on a predefined set of chromosomes, starts and ends. 
        """
        labels = generate_node_labels(self.metadata, self.chr1, self.start1, self.end1, self.chr2, self.start2, self.end2)
        
        # Assert the length of labels equals to total number of bins
        self.assertEqual(len(labels), self.metadata['end'].sum() - self.metadata['start'].sum() + len(self.metadata))

        # Assert that the correct labels are assigned for bins within the RBT region
        self.assertTrue(all(l == -2 for l in labels[120:151]))  # RBT region in chr2 (label=-2)
        self.assertTrue(all(l == -4 for l in labels[340:371]))  # RBT region in chr4 (label=-4)

        # Assert that the correct labels are assigned for bins outside the RBT region
        self.assertTrue(all(l == 1 for l in labels[0:100]))     # chr1 (label=1)
        self.assertTrue(all(l == 2 for l in labels[100:120]))   # chr2 before RBT region (label=2)
        self.assertTrue(all(l == 2 for l in labels[151:200]))   # chr2 after RBT region (label=2)
        self.assertTrue(all(l == 3 for l in labels[200:300]))   # chr3 (label=3)
        self.assertTrue(all(l == 4 for l in labels[300:340]))   # chr4 before RBT region (label=4)
        self.assertTrue(all(l == 4 for l in labels[371:400]))   # chr4 after RBT region (label=4)
        self.assertTrue(all(l == 5 for l in labels[400:500]))   # chr5 (label=5)

class TestEdgeCases(unittest.TestCase):
    def test_start_larger_than_end(self):
        """
        Test that an error is raised when `start` is larger than `end` in `metadata`.
        """
        metadata = pd.DataFrame({
            'chr': ['chr1', 'chr2'],
            'start': [100, 200],
            'end': [99, 199]
        })
        with self.assertRaises(ValueError):
            get_translocated_bins(metadata, 'chr1', 10, 20, 'chr2', 130, 140)

    def test_non_standard_chr_names(self):
        """
        Test that an error is raised when non-standard chromosome names are present.
        """
        metadata = pd.DataFrame({
            'chr': ['chr1A', 'chr2B'],
            'start': [0, 100],
            'end': [99, 199]
        })
        with self.assertRaises(ValueError):
            get_translocated_bins(metadata, 'chr1', 10, 20, 'chr2', 130, 140)

    def test_overlapping_ranges(self):
        """
        Test that an error is raised when overlapping ranges are present in `metadata`.
        """
        metadata = pd.DataFrame({
            'chr': ['chr1', 'chr2', 'chr1'],
            'start': [0, 100, 50],
            'end': [99, 199, 150]
        })
        with self.assertRaises(ValueError):
            get_translocated_bins(metadata, 'chr1', 10, 20, 'chr2', 130, 140)

    def test_na_values_in_metadata(self):
        """
        Test that an error is raised when `metadata` contains NA values.
        """
        metadata = pd.DataFrame({
            'chr': ['chr1', 'chr2', np.nan],
            'start': [0, 100, np.nan],
            'end': [99, 199, np.nan]
        })
        with self.assertRaises(ValueError):
            get_translocated_bins(metadata, 'chr1', 10, 20, 'chr2', 130, 140)

    def test_negative_values_in_metadata(self):
        """
        Test that an error is raised when `start` or `end` values are negative in `metadata`.
        """
        metadata = pd.DataFrame({
            'chr': ['chr1', 'chr2'],
            'start': [-10, 100],
            'end': [99, -50]
        })
        with self.assertRaises(ValueError):
            get_translocated_bins(metadata, 'chr1', 10, 20, 'chr2', 130, 140)




if __name__ == '__main__':
    unittest.main()

