import unittest
import pandas as pd
import sys
sys.path.append('..')
from translocated_data import generate_node_labels

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

if __name__ == '__main__':
    unittest.main()

