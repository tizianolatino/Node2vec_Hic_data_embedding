def add_rbt_to_data(data, metadata, chr1, start1, end1, chr2, start2, end2):
    
    """ to implement """
    
    return data


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
