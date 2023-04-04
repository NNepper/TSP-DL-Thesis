import torch
import torch.nn as nn

from model.encoder import GNNEncoder
from model.decoder import DotDecoder


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


class Graph2Graph(torch.nn.Module):
    def __init__(self, graph_size, hidden_dim=100, num_layers=4, num_heads=1):
        super().__init__()
        self.encoder = GNNEncoder(hidden_dim, num_layers, num_heads)

        self.decoder = DotDecoder(graph_size)

    def forward(self, x, edge_index, edge_attributes):
        z = self.encoder(x, edge_index, edge_attributes)
        pi = self.decoder(z, edge_index)
        return pi
