import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GNNEncoder(nn.Module):
    def __init__(self, hidden_dim, drop_rate=0.0, heads=4, layer_number=4):
        super().__init__()
        self.dropout = drop_rate
        self.gnn = GATv2Conv(in_channels=2, out_channels=hidden_dim, heads=heads, edge_dim=1, jk="lstm",
                             num_layers=layer_number)
        self.bnorm = nn.BatchNorm1d(hidden_dim * heads)
        self.activ = nn.ReLU()

    def forward(self, x, edge_index, edge_attributes):
        x = self.gnn(x, edge_index, edge_attributes)
        x = self.bnorm(x)
        x = self.activ(x)
        return x