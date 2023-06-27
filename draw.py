import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def plot_data(data, metadata, colorbar_boundaries=None):
    """
    Plots the data using metadata to label the chromosomes.

    Args:
    data (pd.DataFrame): The adjacency matrix represented as a dataframe.
    metadata (pd.DataFrame): A dataframe containing the metadata for chromosomes. 
                             It must have the columns "chr", "start", and "end".
    colorbar_boundaries (list of float, optional): Boundaries for the colors in the colorbar.

    Returns:
    None
    """
    fig, ax = plt.subplots()

    # If colorbar_boundaries is provided, use it to define the colormap
    if colorbar_boundaries is not None:
        cmap = plt.get_cmap('viridis', len(colorbar_boundaries)-1)
        norm = colors.BoundaryNorm(colorbar_boundaries, cmap.N)
        im = ax.imshow(data, cmap=cmap, norm=norm, interpolation='none')
    else:
        im = ax.imshow(data, cmap='viridis', interpolation='none')

    # Calculate the tick positions and labels
    tick_positions = []
    tick_labels = []
    for _, row in metadata.iterrows():
        center = (row['start'] + row['end']) // 2
        tick_positions.append(center)
        tick_labels.append(row['chr'])

    # Set the ticks
    ax.set_yticks(tick_positions)

    # Set the tick labels
    ax.set_yticklabels(tick_labels)

   

    # Add a color bar
    cbar = plt.colorbar(im, ax=ax, pad=0.01)

    plt.show()



def plot_embedding(embedding, node_labels):
    """
    Plots a 2D embedding with different colors for different chromosomes 
    and different alpha for RBT nodes.

    Args:
    embedding (np.ndarray): The 2D embedded data, shape (n_nodes, 2).
    node_labels (np.ndarray): An array of labels for the nodes.
    """

    # Create a color map
    cmap = plt.get_cmap('tab20')

    # Get the unique node labels and assign a color to each one
    unique_labels = np.unique(node_labels)
    colors = cmap(np.linspace(0, 1, len(unique_labels)))

    # Create a mapping from labels to colors
    label_to_color = dict(zip(unique_labels, colors))

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Iterate over the unique labels
    for label in unique_labels:
        # Get the nodes with this label
        nodes = np.where(node_labels == label)[0]

        # If the label is negative (corresponds to an RBT region), use a lower alpha
        if label < 0:
            alpha = 0.2
            chr_label = f'tchr{abs(label)}'
        else:
            alpha = 1.0
            chr_label = f'chr{abs(label)}'

        # Scatter the nodes
        ax.scatter(embedding[nodes, 0], embedding[nodes, 1], 
                   color=label_to_color[label], alpha=alpha, label=chr_label)

    # Add a legend and show the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=2)
    plt.show()
