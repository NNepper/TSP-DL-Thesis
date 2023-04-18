import torch.nn as nn
from torch_geometric.nn import PointNetConv, GATv2Conv


class GATEncoder(nn.Module):
    def __init__(self, hidden_dim, drop_rate=0.0, num_layers=4, num_heads=4):
        super().__init__()
        self.dropout = drop_rate
        self.gnn = GATv2Conv(in_channels=2, out_channels=hidden_dim // num_heads, heads=num_heads, edge_dim=1, jk="lstm",
                             num_layers=num_layers)
        self.bnorm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.gnn(x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bnorm(x)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, hidden_dim, drop_rate=0.0, num_layers=4, num_worker=16):
        super().__init__()
        self.dropout = drop_rate
        self.conv1 = PointNetConv(
            local_nn=nn.Sequential(
                nn.Linear(2 + 2, 64),
                nn.LeakyReLU(negative_slope=.05),
                nn.Linear(64, 128),
                nn.LeakyReLU(negative_slope=.05),
            ),
            global_nn=nn.Sequential(
                nn.Linear(128, 256),
                nn.LeakyReLU(negative_slope=.05),
                nn.Linear(256, 512),
                nn.LeakyReLU(negative_slope=.05),
            ),
        )
        self.bnorm1 = nn.BatchNorm1d(512)

        self.conv2 = PointNetConv(
            local_nn=nn.Sequential(
                nn.Linear(512 + 2, 1024),
                nn.LeakyReLU(negative_slope=.05),
                nn.Linear(1024, 1024),
                nn.LeakyReLU(negative_slope=.05),
            ),
            global_nn=nn.Sequential(
                nn.Linear(1024, 2048),
                nn.LeakyReLU(negative_slope=.05),
                nn.Linear(2048, 2048),
                nn.LeakyReLU(negative_slope=.05),
            ),
        )
        self.bnorm2 = nn.BatchNorm1d(2048)
        self.activ = nn.LeakyReLU(negative_slope=0.05)



    def forward(self, x, edge_index, edge_attribute):
        pos = x

        x = self.conv1(x, pos, edge_index)
        x = self.activ(x)
        x = self.bnorm1(x)

        x = self.conv2(x, pos, edge_index)
        x = self.activ(x)
        x = self.bnorm2(x)

        return x