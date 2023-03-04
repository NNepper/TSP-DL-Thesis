import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import softmax


def custom_loss(pred_pi, pairwise_distance, opt_length):
    expected_length = .0
    traversed_nodes = {}
    curr_idx = 0
    for _ in range(pairwise_distance.shape[0]):
        # Select most_probable next node
        next_idx = torch.max(pairwise_distance[curr_idx, :])
        expected_length += pred_pi[curr_idx, next_idx] * pairwise_distance
    return .0


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
        dataset = pickle.load(f)
        data, target, opt_length = dataset[0]

    # Model Initialization
    model = Graph2Graph(input_dim=10, hidden_dim=64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    total_loss = total_examples = prev_loss = 0
    for epoch in range(1, 100000):
        optimizer.zero_grad()

        data.x = data.x.to(torch.float32)
        data.to(device)
        pred = model.forward(data)

        loss = custom_loss(pred, target)

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