import pickle
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import softmax
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data

def custom_loss(pred_pi, pairwise_distance, opt_length):
    expected_length = torch.Tensor([.0])
    traversed_nodes = set()
    curr_idx = 0
    route = torch.argmax(pairwise_distance, dim=1)
    for i, next_idx in enumerate(route):
        expected_length += pred_pi[curr_idx, next_idx] * pairwise_distance[curr_idx, next_idx]
        traversed_nodes |= {next_idx}
    if len(traversed_nodes) < pairwise_distance.shape[0]: # Penalize non-Hamiltonian tour
        penalty = (len(traversed_nodes) - pairwise_distance) * opt_length
        return torch.abs((expected_length + penalty) - opt_length)
    return torch.abs(expected_length - opt_length)


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_rate=0.1):
        super().__init__()
        self.dropout = drop_rate
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.activ = nn.ReLU()
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.activ(x)

        x = self.conv2(x, edge_index)
        x = self.activ(x)
        return x


class DotDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax()

    def forward(self, x, edge_index):
        logit = x @ x.t()
        probs = self.softmax(logit)
        return probs


class Graph2Graph(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = GNNEncoder(input_dim, hidden_dim)
        self.decoder = DotDecoder()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.encoder(data)
        pi = self.decoder(z, edge_index)
        return pi


if __name__ == '__main__':
    # Data importing
    with open('data/dataset.pkl', 'rb') as f:
        graphs, target, opt_length = pickle.load(f)
        dataLoader = DataLoader(graphs, batch_size=2)

    # Model Initialization
    model = Graph2Graph(input_dim=10, hidden_dim=64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    total_loss = total_examples = prev_loss = 0
    for batch in tqdm.tqdm(dataLoader):
        for epoch in range(1, 100000):
            optimizer.zero_grad()

            batch.x = batch.x.to(torch.float32)
            batch.to(device)
            pred = model.forward(batch)

            loss = custom_loss(pred, batch.x, opt_length)

            loss.backward()
            optimizer.step()
            total_loss += loss.sum()
            total_examples += pred.numel()
            if epoch % 100 == 0:
                print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
                if abs(total_loss - prev_loss) > 10e-6:
                    prev_loss = total_loss
                else:
                    print(pred)
                    break