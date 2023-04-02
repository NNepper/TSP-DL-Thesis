import torch
import torch.nn as nn

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax


class DotDecoder(nn.Module):
    def __init__(self, graph_size):
        super().__init__()
        self.graph_size = graph_size

    def forward(self, x, edge_index):
        """
        The forward function takes in a batch of node features and returns the
        probability distribution over all possible edges. The probability is computed
        by taking the softmax of the dot product between each pair of nodes. This is
        equivalent to computing a similarity score for each pair, and then normalizing
        the scores so that they sum to 1.

        :param self: Access variables that belong to the class
        :param x: Pass the node features to the forward function
        :param edge_index: Construct the adjacency matrix
        :return: A tensor of size (batch_size, graph_size, graph_size)
        """
        pi = torch.zeros(x.shape[0] // self.graph_size, self.graph_size, self.graph_size)
        for i, (x_batch, edge_idx_batch) in enumerate(zip(torch.split(x, self.graph_size), torch.split(edge_index,
                                                                                                       self.graph_size * (
                                                                                                               self.graph_size - 1),
                                                                                                       dim=1)
                                                          )
                                                      ):
            logit = x_batch @ x_batch.t()

            # Compute softmax normalized for each node
            pi[i, :, :] = softmax(
                src=logit.view(self.graph_size * self.graph_size),
                index=torch.arange(0, self.graph_size).repeat(self.graph_size).to(torch.long)
            ).view(self.graph_size, self.graph_size)
        return pi


class ConvNetLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, output_dim):
        super().__init__()
        # Edge attention layers
        self.lin1 = nn.Linear(edge_dim, output_dim)  # edge feature attention
        self.lin2 = nn.Linear(node_dim, output_dim)  # origin feature attention
        self.lin3 = nn.Linear(node_dim, output_dim)  # destination feature attention

        # Embedding layers
        self.lin4 = nn.Linear(node_dim, output_dim)  # origin node embedding
        self.lin5 = nn.Linear(node_dim, output_dim)  # destination node embedding

        self.activation = nn.ReLU()

    def forward(self, x_node, x_edge, edge_index):
        return

class GNNEncoder(nn.Module):
    def __init__(self, hidden_dim, drop_rate=0.0, heads=4, layer_number=4):
        super().__init__()
        self.dropout = drop_rate
        self.gnn = GATv2Conv(in_channels=2, out_channels=hidden_dim, heads=heads, edge_dim=1, jk="lstm",
                             num_layers=layer_number, share_weight=True)
        self.bnorm = nn.BatchNorm1d(hidden_dim * heads)
        self.activ = nn.ReLU()

    def forward(self, x, edge_index, edge_attributes):
        x = self.gnn(x, edge_index, edge_attributes)
        x = self.bnorm(x)
        x = self.activ(x)
        return x


class Graph2Graph(torch.nn.Module):
    def __init__(self, graph_size, hidden_dim=100, num_layers=4, num_heads=1):
        super().__init__()
        self.encoder = GNNEncoder(hidden_dim, num_layers, num_heads)

        self.decoder = DotDecoder(graph_size)

    def forward(self, x, edge_index, edge_attributes):
        z = self.encoder(x, edge_index, edge_attributes)
        pi = self.decoder(z, edge_index)
        return pi
