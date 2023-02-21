import torch.nn as nn
import torch_geometric.nn


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Dense-Layer
        self.conv1 = torch_geometric.nn.SAGEConv(input_dim, hidden_dim)
        self.conv2 = torch_geometric.nn.SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x


class EdgeDecoder(nn.Module):
    def __init__(self, hidden_dim):
        self.lin1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = self.GNNEncoder(data)


class Graph2Graph(nn.Module):
    def __init__(self):
        super().__init__()
