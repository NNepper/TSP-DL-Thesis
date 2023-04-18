import torch
import torch.nn as nn

from model.encoder import GNNEncoder
from model.decoder import DotDecoder

import torch
import torch.nn as nn

from model.encoder import GNNEncoder
from model.decoder import DotDecoder


class Graph2Graph(torch.nn.Module):
    def __init__(self, graph_size, hidden_dim=100, num_layers=4):
        super().__init__()
        self.encoder = GNNEncoder(hidden_dim, num_layers)

        self.decoder = DotDecoder(graph_size)

    def forward(self, x, edge_index, edge_attribute):
        z = self.encoder(x, edge_index, edge_attribute)
        pi = self.decoder(z, edge_index)
        return pi
