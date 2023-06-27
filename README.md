
# Node2vec on Hi-c Data for translocation detection

## Description

Project Name is a Python library that transforms Hi-c data into a Node2Vec embedding, and then applies PCA for dimension reduction. It is useful in graph-based machine learning tasks and provides a convenient way to extract meaningful features from graph-structured data.
This project also provides a set of tools and functions for the analysis and manipulation of Hi-C data. Hi-C is a high-throughput method to investigate the three-dimensional architecture of genomes. These tools help researchers generate a toy-model representation of the Hi-C data using a Negative Binomial distribution. In addition, the toolset allows the introduction of a Reciprocal Balanced Translocation (RBT) into the data, enabling users to simulate and study various genomic rearrangements.

## Installation

To install this project, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/tizianolatino/SC_exam.git
```

2. Navigate into the project directory:

```bash
cd SC_exam
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

from negative_binomial_model import generate_data_from_neg_binomial
from Hic_data_prepro import drop_chr, remove_isolated_bins
from utils import reduce_dimension
from utils_data_metadata import adjust_indices_to_zero
from draw import plot_embedding, plot_data
from translocated_data import generate_node_labels
from node2vec_hic_toydataset import data_to_embedding


# Load the metadata and data
data = pd.read_csv('Data/raw_GM12878_1Mb.csv', header=None)
metadata = pd.read_csv('Data/metadata.csv')


# Drop the specified chromosome and adjust the start and end points of the remaining chromosomes
metadata, data = drop_chr(metadata, data, "'chrY'")

# Remove isolated bins and adjust the start and end points of the chromosomes
metadata, data = remove_isolated_bins(metadata, data)

# Reduce dimension
data, metadata = reduce_dimension(metadata, data, 0.9)

# Generate a new dataframe with values from a negative binomial distribution
neg_bin_data = generate_data_from_neg_binomial(metadata, data)

# Generates labels for each bin in the data
node_labels = generate_node_labels(metadata, "'chr1'", 5, 10, "'chr6'", 105,110)

# Plot the Adajacency Matrix
plot_data(neg_bin_data, metadata, [0,10,100,1000,5000,10000])

# Node Embeddings and PCA data reduction
embeddings = data_to_embedding( neg_bin_data)

# Plots a 2D  of the embeddings
plot_embedding(embeddings, node_labels)
```

Remember to replace 'Data/raw_GM12878_1Mb.csv' and 'Data/metadata.csv' with the paths to your own data and metadata files. This script will preprocess your Hi-C data, generate a new dataset from a negative binomial distribution, label the nodes, plot the adjacency matrix, perform Node2Vec and PCA dimension reduction, and finally plot the 2D embeddings of the nodes.

## Contributing
Contributions are welcome. Feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for more details.

## Contact
For questions or feedback, please contact me at tiziano.latino@studio.unibo.it

## Acknowledgements
Thanks to all contributors and users of the project.
