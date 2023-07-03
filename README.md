
# Node2vec on Hi-c Data for translocation detection

## Description

Project Name is a Python library that transforms Hi-c data into a Node2Vec embedding, and then applies PCA for dimension reduction. It is useful in graph-based machine learning tasks and provides a convenient way to extract meaningful features from graph-structured data.
This project also provides a set of tools and functions for the analysis and manipulation of Hi-C data. Hi-C is a high-throughput method to investigate the three-dimensional architecture of genomes. These tools help researchers generate a toy-model representation of the Hi-C data using a Negative Binomial distribution. In addition, the toolset allows the introduction of a Reciprocal Balanced Translocation (RBT) into the data, enabling users to simulate and study various genomic rearrangements.

## Installation

To install this project, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/tizianolatino/Node2vec_Hic_data_embedding.git
```

2. Navigate into the project directory:

```bash
cd Node2vec_Hic_data_embeddingx
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

Please ensure you have Python 3.6 or later installed on your machine.

## Example of Usage

```python

import pandas as pd
import os
import argparse

from negative_binomial_model import generate_data_from_neg_binomial, intrachr_contacts_mean_var, interchr_contacts_mean_var
from Hic_data_prepro import drop_chr, remove_isolated_bins
from utils import reduce_dimension
from utils_data_metadata import adjust_indices_to_zero
from draw import plot_embedding, plot_data
from translocated_data import apply_RBT_modifications, generate_node_labels
from node2vec_hic_toydataset import data_to_embedding

metadata_ori, data_ori = adjust_indices_to_zero(metadata, data)

# Drop the specified chromosome and adjust the start and end points of the remaining chromosomes
metadata_ori, data_ori = drop_chr(metadata_ori, data_ori, "'chrY'")

# Remove isolated bins and adjust the start and end points of the chromosomes
metadata_ori, data_ori = remove_isolated_bins(metadata_ori, data_ori)

# Calculate the mean and variance for each distance within each chromosome, 
# and for interchromosomal interactions
intrachr_means, intrachr_vars = intrachr_contacts_mean_var(metadata_ori, data_ori)
interchr_mean, interchr_var = interchr_contacts_mean_var(metadata_ori, data_ori)


# Generate a new dataframe with values from a negative binomial distribution
neg_bin_data = generate_data_from_neg_binomial(metadata_ori, data_ori, intrachr_means, intrachr_vars,
                                          interchr_mean, interchr_var )

# Reduce dimension
data_red, metadata_red = reduce_dimension(metadata_ori, neg_bin_data, 0.9)

#Introduce rbt translocation on the data
data_toy_trans = apply_RBT_modifications(data_red, metadata_red, "'chr1'", 5, 10, "'chr6'", 105, 110,
                                    intrachr_means, intrachr_vars,interchr_mean, interchr_var)                                                                

# Generates labels for each bin in the data
node_labels = generate_node_labels(metadata_red, "'chr1'", 5, 10, "'chr6'", 105, 110)

# Plot the Adajacency Matrix
plot_data(data_toy_trans, metadata_red, [0, 10, 100, 1000, 5000, 10000])

# Node Embeddings and PCA data reduction
embeddings = data_to_embedding( data_toy_trans)

# Plots a 2D  of the embeddings
plot_embedding(embeddings, node_labels)
   
```

This script will preprocess your Hi-C data, generate a new dataset from a negative binomial distribution, reduce the dimension of the data, introduce rbt translocation on the data and label the nodes, plot the adjacency matrix, perform Node2Vec and PCA dimension reduction, and finally plot the 2D embeddings of the nodes.

## Contributing
Contributions are welcome. Feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for more details.

## Contact
For questions or feedback, please contact me at tiziano.latino@studio.unibo.it

## Acknowledgements
Thanks to all contributors and users of the project.
