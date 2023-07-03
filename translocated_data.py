from itertools import combinations
import numpy as np

def get_translocated_bins(metadata, chr1, start1, end1, chr2, start2, end2):
    """
    Get the list of bins for each chromosome after simulating a reciprocal balanced translocation.

    Args:
    metadata (pd.DataFrame): The metadata for the chromosomes. 
                             It must have the columns "chr", "start", and "end".
    chr1, chr2 (str): The chromosomes involved in the translocation.
    start1, end1, start2, end2 (int): The start and end bins on the respective chromosomes involved in the translocation.

    Returns:
    list, list: The lists of bins for the two chromosomes after the translocation.
    """
    # Get the start and end indices for the chromosomes
    chr1_start, chr1_end = metadata.loc[metadata['chr'] == chr1, ['start', 'end']].values[0]
    chr2_start, chr2_end = metadata.loc[metadata['chr'] == chr2, ['start', 'end']].values[0]

    # Construct the list of bins for each chromosome
    chr1_bins = list(range(chr1_start, start1)) + list(range(start2, end2+1)) + list(range(end1+1, chr1_end+1))
    chr2_bins = list(range(chr2_start, start2)) + list(range(start1, end1+1)) + list(range(end2+1, chr2_end+1))

    return chr1_bins, chr2_bins


def modify_interchr_contacts(data, metadata, chr1, start1, end1, chr2, start2, end2,intrachr_means, intrachr_vars):
    """
    Modifies the contact values in the interchr data for the bins involved in a reciprocal balanced translocation.

    Args:
    data (pd.DataFrame): The original contact data.
    metadata (pd.DataFrame): The metadata for the chromosomes.
                             It must have the columns "chr", "start", and "end".
    chr1, chr2 (str): The chromosomes involved in the translocation.
    start1, end1, start2, end2 (int): The start and end bins on the respective chromosomes involved in the translocation.

    Returns:
    pd.DataFrame: The modified contact data after the translocation.
    """
    
    # Create a copy of the data
    data_copy = data.copy()
    
    # Get the translocated bin lists
    chr1_bins, chr2_bins = get_translocated_bins(metadata, chr1, start1, end1, chr2, start2, end2)


    # Substitute the original contacts with values from a negative binomial distribution
    for chr_bins, chr_name in zip([chr1_bins, chr2_bins], [chr1, chr2]):
        
        prev_p = prev_r = None
        # Generate the new contacts
        for i, j in combinations(chr_bins, 2):
            dist = abs(chr_bins.index(i) - chr_bins.index(j))
            
            # Get mean and variance for this chromosome and distance
            mean_ij = intrachr_means[chr_name].get(dist)
            var_ij = intrachr_vars[chr_name].get(dist)
            
            # Check if mean and variance exist
            if mean_ij is not None and var_ij is not None:
                # Calculate the parameters for the negative binomial distribution
                if var_ij != None and mean_ij != None and var_ij > mean_ij and var_ij != 0:
                    p = mean_ij/var_ij
                    r = mean_ij**2/(var_ij-mean_ij)
                else:
                    p, r = prev_p, prev_r
                prev_p, prev_r = p, r
         
                # Generate contact from the negative binomial distribution
                contact_ij = np.random.negative_binomial(r, p)
            
                # Substitute the new contact in the DataFrame
                data_copy.at[i, j] = contact_ij
                data_copy.at[j, i] = contact_ij  # Ensure the matrix remains symmetric

    return data_copy

def modify_intrachr_contacts(metadata, data, chr1, start1, end1, chr2, start2, end2, interchr_mean, interchr_var):
    """
    Modifies the contact values in the intrachr data for the bins involved in a reciprocal balanced translocation.

    Args:
    metadata (pd.DataFrame): A dataframe containing the metadata for chromosomes. 
                             It must have the columns "chr", "start", and "end".
    data (pd.DataFrame): The adjacency matrix represented as a dataframe, representing contacts between bins.
    chr1, chr2 (str): The chromosomes involved in the RBT.
    start1, end1, start2, end2 (int): The start and end positions (bins) on the respective chromosomes involved in the RBT.

    Returns:
    pd.DataFrame: The dataframe with modified contact values.
    """
    
    # Create a copy of the data
    data_copy = data.copy()

    # Calculate parameters for negative binomial distribution
    r = interchr_mean ** 2 / (interchr_var - interchr_mean)
    p = interchr_mean / interchr_var

    # Put the parameters into a list for easy looping
    chr_params = [(chr1, start1, end1), (chr2, start2, end2)]

    for chr_name, start, end in chr_params:
        # Extract the bins involved in the RBT
        t_bins = list(range(start, end+1))

        # Iterate over the bins
        for bin in t_bins:

            # Update the contacts of bin with all other bins in the chromosome
            for other_bin in range(metadata.loc[metadata['chr'] == chr_name, 'start'].values[0],
                                   metadata.loc[metadata['chr'] == chr_name, 'end'].values[0] + 1):
                if other_bin not in t_bins:
                    
                    # Generate a new contact from the negative binomial distribution for this bin
                    new_contact = np.random.negative_binomial(r, p)
                    
                    # Update the contact matrix in the dataframe
                    data_copy.iloc[bin, other_bin] = new_contact
                    data_copy.iloc[other_bin, bin] = new_contact  # Assuming the contact matrix is symmetric

    return data_copy

def apply_RBT_modifications(data, metadata, chr1, start1, end1, chr2, start2, end2, intrachr_means, intrachr_vars,
                            interchr_mean, interchr_var):
    """
    Applies both intra- and inter-chromosomal modifications due to a reciprocal balanced translocation.

    Args:
    data (pd.DataFrame): The original contact data.
    metadata (pd.DataFrame): The metadata for the chromosomes.
                             It must have the columns "chr", "start", and "end".
    chr1, chr2 (str): The chromosomes involved in the translocation.
    start1, end1, start2, end2 (int): The start and end bins on the respective chromosomes involved in the translocation.

    Returns:
    pd.DataFrame: The modified contact data after the translocation.
    """
    # Create a copy of the data
    data_copy = data.copy()
    
    # Modify the intra-chromosomal contacts
    data_copy = modify_intrachr_contacts(metadata, data_copy, chr1, start1, end1, chr2, start2, end2,interchr_mean, interchr_var)

    # Modify the inter-chromosomal contacts
    data_copy = modify_interchr_contacts(data_copy, metadata, chr1, start1, end1, chr2, start2, end2, intrachr_means, intrachr_vars)

    return data_copy

def generate_node_labels(metadata, chr1, start1, end1, chr2, start2, end2):
    """
    Generates labels for each bin in the data.

    Args:
    metadata (pd.DataFrame): A dataframe containing the metadata for chromosomes. 
                             It must have the columns "chr", "start", and "end".
    chr1, chr2 (str): The chromosomes involved in the RBT.
    start1, end1, start2, end2 (int): The start and end positions (bins) on the respective chromosomes involved in the RBT.

    Returns:
    list: A list of labels for each bin.
    """
    labels = []
    chr1_start, chr1_end = metadata.loc[metadata['chr'] == chr1, ['start', 'end']].values[0]
    chr2_start, chr2_end = metadata.loc[metadata['chr'] == chr2, ['start', 'end']].values[0]
    label_counter = 1

    # Assign labels to each bin
    for _, row in metadata.iterrows():
        if row['chr'] == chr1:
            # For bins in chr1
            labels.extend([label_counter] * (start1 - chr1_start))
            labels.extend([-label_counter] * (end1 - start1 + 1))  # Assign -label_counter for RBT region
            labels.extend([label_counter] * (chr1_end - end1))
        elif row['chr'] == chr2:
            # For bins in chr2
            labels.extend([label_counter] * (start2 - chr2_start))
            labels.extend([-label_counter] * (end2 - start2 + 1))  # Assign -label_counter for RBT region
            labels.extend([label_counter] * (chr2_end - end2))
        else:
            # For bins in other chromosomes
            labels.extend([label_counter] * (row['end'] - row['start'] + 1))
        label_counter += 1

    return labels
