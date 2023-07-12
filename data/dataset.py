import torch 
from torch.utils.data import Dataset

import numpy as np

from scipy.spatial.distance import pdist, squareform

def nearest_neighbor_graph(nodes, neighbors, knn_strat):
    """Returns k-Nearest Neighbor graph as a **NEGATIVE** adjacency matrix
    """
    num_nodes = len(nodes)
    # If `neighbors` is a percentage, convert to int
    if knn_strat == 'percentage':
        neighbors = int(num_nodes * neighbors)
    
    if neighbors >= num_nodes-1 or neighbors == -1:
        W = np.zeros((num_nodes, num_nodes))
    else:
        # Compute distance matrix
        W_val = squareform(pdist(nodes, metric='euclidean'))
        W = np.ones((num_nodes, num_nodes))
        
        # Determine k-nearest neighbors for each node
        knns = np.argpartition(W_val, kth=neighbors, axis=-1)[:, neighbors::-1]
        # Make connections
        for idx in range(num_nodes):
            W[idx][knns[idx]] = 0
    
    # Remove self-connections
    np.fill_diagonal(W, 1)
    return W


def tour_nodes_to_W(tour_nodes):
    """Computes edge adjacency matrix representation of tour
    """
    num_nodes = len(tour_nodes)
    tour_edges = np.zeros((num_nodes, num_nodes))
    for idx in range(len(tour_nodes) - 1):
        i = tour_nodes[idx]
        j = tour_nodes[idx + 1]
        tour_edges[i][j] = 1
        tour_edges[j][i] = 1
    # Add final connection
    tour_edges[j][tour_nodes[0]] = 1
    tour_edges[tour_nodes[0]][j] = 1
    return tour_edges
    
class TSPDataset(Dataset):
    def __init__(self, filename=None, batch_size=128, neighbors=-1, 
                 knn_strat=None):
        """Class representing a PyTorch dataset of TSP instances, which is fed to a dataloader

        Args:
            filename: File path to read from (for SL)
            batch_size: Batch size for data loading/batching
            neighbors: Number of neighbors for k-NN graph computation (-1 for complete graph)
            knn_strat: Strategy for computing k-NN graphs ('percentage'/'standard')
        """
        super(TSPDataset, self).__init__()

        self.filename = filename
        self.batch_size = batch_size
        self.knn_strat = knn_strat

        # Loading from file (usually used for Supervised Learning or evaluation)
        if filename is not None:
            self.nodes_coords = [] 
            self.tour_nodes = []

            print('\nLoading from {}...'.format(filename))
            for line in open(filename, "r").readlines():
                line = line.split(" ")
                num_nodes = int(line.index('output')//2)
                self.nodes_coords.append(
                    [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
                )
                tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
                self.tour_nodes.append(tour_nodes)

        self.size = len(self.nodes_coords)
        self.num_nodes = len(self.nodes_coords[0])
        if neighbors == -1:
            self.neighbors = self.num_nodes - 1
        else:
            self.neighbors = neighbors
            assert self.neighbors <= self.num_nodes -1, \
                "Number of neighbors ({}) must be less than number of nodes ({})".format(self.neighbors, self.size)
        assert self.size % batch_size == 0, \
            "Number of samples ({}) must be divisible by batch size ({})".format(self.size, batch_size)
 
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        tour_nodes = self.tour_nodes[idx]
        return torch.FloatTensor(self.nodes_coords[idx]), torch.LongTensor(tour_nodes), 