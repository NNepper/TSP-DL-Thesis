import torch
import torch.nn as nn

from torch_geometric.nn import GATConv, InnerProductDecoder

class DotDecoder(nn.Module):
    def __init__(self, graph_size):
        super().__init__()
        self.softmax = nn.Softmax()
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
        for i, x_batch in enumerate(torch.split(x, self.graph_size)):
            logit = x_batch @ x_batch.t()
            # Compute softmax over whole edges

            pi[i, :] = self.softmax((x_batch @ x_batch.t()).view(self.graph_size * self.graph_size)).view(
                self.graph_size, self.graph_size)
        return pi


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_rate=0.0):
        super().__init__()
        self.dropout = drop_rate
        self.gnn1 = GATConv(input_dim, hidden_dim, head=3)
        self.gnn2 = GATConv(hidden_dim, hidden_dim, head=3)
        self.gnn3 = GATConv(hidden_dim, hidden_dim, head=3)
        self.gnn4 = GATConv(hidden_dim, hidden_dim, head=3)
        self.activ = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.gnn1(x, edge_index)
        x = self.activ(x)

        x = self.gnn2(x, edge_index)
        x = self.activ(x)

        x = self.gnn3(x, edge_index)
        x = self.activ(x)

        x = self.gnn4(x, edge_index)
        x = self.activ(x)

        return x


class Graph2Graph(torch.nn.Module):
    def __init__(self, graph_size, hidden_dim=100):
        super().__init__()
        self.encoder = GNNEncoder(graph_size + 2, hidden_dim)
        self.decoder = InnerProductDecoder()

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        pi = self.decoder(z, edge_index, sigmoid=True)
        return pi
