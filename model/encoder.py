import torch.nn as nn
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    def __init__(self, hidden_dim, drop_rate=0.0, num_layers=4, num_heads=4):
        super().__init__()
        self.dropout = drop_rate
        self.gnn = GATConv(in_channels=2, out_channels=hidden_dim // num_heads, heads=num_heads, edge_dim=1,
                             num_layers=num_layers)

    def forward(self, x, edge_index, edge_attr):
        x = self.gnn(x, edge_index=edge_index, edge_attr=edge_attr)
        return x