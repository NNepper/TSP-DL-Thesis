import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GNNEncoder(nn.Module):
    def __init__(self, hidden_dim, drop_rate=0.1, layer_number=4, heads=4):
        super().__init__()
        self.dropout = drop_rate
        self.gnn = GATv2Conv(in_channels=2, out_channels=hidden_dim, heads=heads, edge_dim=1, jk="lstm",
                             num_layers=layer_number, drop_rate=drop_rate)
        self.bnorm = nn.BatchNorm1d(hidden_dim * heads)

    def forward(self, x, edge_index, edge_attr):
        x = self.gnn(x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bnorm(x)
        return x


class MyGNNEncoder(nn.Module):
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
