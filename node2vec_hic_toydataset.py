import networkx as nx
from node2vec import Node2Vec
from sklearn.decomposition import PCA

def data_to_embedding(data, n2v_dimensions=16, dimensions=2, walk_length=30, num_walks=100, workers=1, p=1, q=1):
    """
    Transforms adjacency data to a Node2Vec embedding, and then applies PCA for dimension reduction.

    Args:
    data (pd.DataFrame): The adjacency matrix data.
    n2v_dimensions (int): The number of dimensions for Node2Vec.
    dimensions (int): The number of dimensions for PCA reduction.
    walk_length (int): The length of each walk for Node2Vec.
    num_walks (int): The number of walks per node for Node2Vec.
    workers (int): The number of workers for parallel computation.
    p (float): Node2Vec return hyperparameter.
    q (float): Node2Vec in-out hyperparameter.

    Returns:
    np.ndarray: The data embedded into the specified number of dimensions.
    """
    
    if n2v_dimensions <= 0:
        raise ValueError("n2v_dimensions must be greater than 0")
    if walk_length <= 0:
        raise ValueError("walk_length must be greater than 0")
    if dimensions <= 0:
        raise ValueError("PCA dimensions must be greater than 0")
    if p < 0 or q < 0:
        raise ValueError("Parameters p and q must be non-negative")
    
    # Convert the DataFrame to a numpy array
    adjacency = data.values

    # Create a NetworkX graph from the adjacency matrix
    graph = nx.from_numpy_array(adjacency)

    # Initialize Node2Vec
    node2vec = Node2Vec(graph, dimensions=n2v_dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, p=p, q=q)

    # Fit and transform Node2Vec
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node2vec_embeddings = model.wv.vectors

    # Apply PCA for dimension reduction
    pca = PCA(n_components=dimensions)
    reduced_embeddings = pca.fit_transform(node2vec_embeddings)

    return reduced_embeddings
