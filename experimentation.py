import pickle
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AGNNConv
from torch_geometric.utils import softmax
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data

from common.utils import sample_draw_probs_graph


def custom_loss(batch_pi, batch_distances, opt_length):
    expected_length = torch.zeros(batch_pi.shape[0])
    traversed_nodes = set()
    pairwise_distances = torch.split(batch_distances, batch_distances.shape[0] // opt_length.shape[0])
    curr_idx = 0
    for i, pred in enumerate(batch_pi):
        route = torch.argmax(pairwise_distances[i], dim=1)
        for next_idx in route:
            expected_length += pred[curr_idx, next_idx] * pairwise_distances[i][curr_idx, next_idx]
            traversed_nodes |= {int(next_idx)}
        if len(traversed_nodes) < pairwise_distances[i].shape[0]:  # Penalize non-Hamiltonian tour
            penalty = (len(traversed_nodes) - pairwise_distances[i].shape[0]) * opt_length
            return torch.abs((expected_length + penalty) - opt_length)
    return torch.abs(expected_length - opt_length)


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_rate=0.10):
        super().__init__()
        self.dropout = drop_rate
        self.conv1 = AGNNConv(input_dim, hidden_dim)
        self.activ = nn.ReLU()
        self.conv2 = AGNNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activ(x)

        x = self.conv2(x, edge_index)
        x = self.activ(x)
        return x


class DotDecoder(nn.Module):
    def __init__(self, graph_size):
        super().__init__()
        self.softmax = nn.Softmax()
        self.graph_size = graph_size

    def forward(self, x, edge_index):
        pi = torch.zeros(x.shape[0] // self.graph_size, self.graph_size, self.graph_size)
        for i, x_batch in enumerate(torch.split(x, self.graph_size)):
            logit = x_batch @ x_batch.t()
            pi[i, :] = self.softmax(logit)
        return pi


class Graph2Graph(torch.nn.Module):
    def __init__(self, graph_size, hidden_dim):
        super().__init__()
        self.encoder = GNNEncoder(graph_size, hidden_dim)
        self.decoder = DotDecoder(graph_size)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        pi = self.decoder(z, edge_index)
        return pi


if __name__ == '__main__':
    # Data importing
    with open('data/dataset.pkl', 'rb') as f:
        graphs, target, opt_length = pickle.load(f)
        dataLoader = DataLoader(graphs, batch_size=100)

    # Model Initialization
    model = Graph2Graph(graph_size=10, hidden_dim=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    total_loss = total_examples = prev_loss = 0
    for batch in tqdm.tqdm(dataLoader):
        for epoch in range(1, 100000):
            optimizer.zero_grad()

            batch.x = batch.x.to(torch.float32)
            batch.to(device)
            pred = model.forward(batch.x, batch.edge_index)

            loss = custom_loss(pred, batch.x, batch.y).sum()

            loss.backward()
            optimizer.step()
            total_loss += loss.sum()
            total_examples += pred.numel()
            if epoch % 50 == 0:
                print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
                if abs(total_loss - prev_loss) > 10e-6:
                    prev_loss = total_loss
                else:
                    fig, axs = sample_draw_probs_graph(batch, pred)
                    fig.show()
                    break
