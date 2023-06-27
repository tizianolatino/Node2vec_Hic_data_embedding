from scipy.stats import nbinom
import numpy as np

def assertAlmostEqualRelative(self, a, b, rel_tol=1e-2):
    """
    Checks if 'a' and 'b' are approximately equal relative to the magnitude of 'a'.

    Args:
    a, b: The two numbers to compare.
    rel_tol: The relative tolerance -- the maximum allowed difference between 'a' and 'b',
             relative to the magnitude of 'a'.
    """
    diff = abs(a - b)
    self.assertLessEqual(diff, abs(rel_tol * a))
    
    
def validate_neg_binomial_distribution(metadata, original_data, neg_bin_data):
    """
    This function validates that data generated from a negative binomial distribution matches the original data.

    Args:
        metadata (pd.DataFrame): A DataFrame with metadata about the data.
        original_data (pd.DataFrame): A DataFrame with the original data.
        neg_bin_data (pd.DataFrame): A DataFrame with data generated from a negative binomial distribution.

    Returns:
        bool: Returns True if the original data matches the distribution of the generated data, 
        otherwise it raises an AssertionError.
    """

    for _, row in metadata.iterrows():
        for distance in range(row['end'] - row['start'] + 1):
            if distance == 0 or distance >10: continue
            # Select the data for this distance
            original_values = original_data.loc[row['start']:row['end'], row['start']:row['end']].iloc[::distance].values.flatten()
            generated_values = neg_bin_data.loc[row['start']:row['end'], row['start']:row['end']].iloc[::distance].values.flatten()

            # Calculate the mean and variance for the original and generated values
            mean_original = original_values.mean()
            var_original = original_values.var()
            mean_generated = generated_values.mean()
            var_generated = generated_values.var()

            # Check that the parameters are valid for a negative binomial distribution
            if mean_original > 0 and var_original > mean_original:
                p_original = mean_original/var_original
                r_original = mean_original**2/(var_original-mean_original)
                mean_nbinomial = nbinom.mean(r_original, p_original)
                var_nbinomial = nbinom.var(r_original, p_original)
                assert np.isclose(mean_nbinomial, mean_generated, rtol=0.5), \
                       "Mean of generated data does not match expected mean from original data"
                assert np.isclose(var_nbinomial, var_generated, rtol=0.5), \
                       "Variance of generated data does not match expected variance from original data"
            else:
                raise AssertionError("The original data does not have valid parameters for a negative binomial distribution")

    return True
