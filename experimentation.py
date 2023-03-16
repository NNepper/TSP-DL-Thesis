import pickle
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data

from common.utils import sample_draw_probs_graph


def entropy_mixed_loss(batch_pi, batch_distances, opt_length):
    loss = torch.zeros(opt_length.shape)
    pairwise_distances = torch.split(batch_distances, batch_distances.shape[0] // opt_length.shape[0])
    for i, pred in enumerate(batch_pi):
        # Greedy Decoding of the tour
        curr_idx = 0
        pred_length = .0
        entropy_sum = .0
        pi_cpy = torch.clone(pred)
        for _ in range(len(opt_length) - 1):
            next_idx = torch.argmax(pi_cpy[curr_idx, :])
            pred_length += pairwise_distances[i][curr_idx, next_idx]
            entropy_sum += torch.log(pred[curr_idx, next_idx])
            pi_cpy[:, next_idx] = .0
            curr_idx = next_idx
        loss[i] = pred_length - opt_length[i] - entropy_sum
    return loss


def policy_gradient_loss(batch_pi, batch_distances, opt_length):
    loss = torch.zeros(opt_length.shape)
    pairwise_distances = torch.split(batch_distances, batch_distances.shape[0] // opt_length.shape[0])
    for i, pred in enumerate(batch_pi):
        # Greedy Decoding of the tour
        curr_idx = 0
        pred_length = .0
        entropy_sum = .0
        for _ in range(len(opt_length) - 1):
            next_idx = torch.argmax(pred[curr_idx, :])
            loss[i] += torch.log(pred[curr_idx, next_idx]) * pairwise_distances[i][curr_idx, next_idx]
            curr_idx = next_idx
        loss[i] = pred_length - opt_length[i]
    return loss


def custom_loss(batch_pi, batch_distances, opt_length):
    expected_length = torch.zeros(batch_pi.shape[0])
    traversed_nodes = set()
    pairwise_distances = torch.split(batch_distances, batch_distances.shape[0] // opt_length.shape[0])
    for i, pred in enumerate(batch_pi):
        route = torch.argmax(pairwise_distances[i], dim=1)
        curr_idx = 0
        for next_idx in route:
            expected_length += pred[curr_idx, next_idx] * pairwise_distances[i][curr_idx, next_idx]
            traversed_nodes |= {int(next_idx)}
            curr_idx = next_idx
        if len(traversed_nodes) < pairwise_distances[i].shape[0]:  # Penalize non-Hamiltonian tour
            penalty = (len(traversed_nodes) - pairwise_distances[i].shape[0]) * opt_length
            return torch.abs((expected_length + penalty) - opt_length)
    return torch.abs(expected_length - opt_length)


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_rate=0.5):
        super().__init__()
        self.dropout = drop_rate
        self.conv1 = GATConv(input_dim, hidden_dim, head=4)
        self.activ = nn.ReLU()
        self.conv2 = GATConv(hidden_dim, hidden_dim, head=4)

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
        self.encoder = GNNEncoder(graph_size + 2, hidden_dim)
        self.decoder = DotDecoder(graph_size)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        pi = self.decoder(z, edge_index)
        return pi


if __name__ == '__main__':
    # Data importing
    with open('data/dataset_10.pkl', 'rb') as f:
        graphs, target, opt_length = pickle.load(f)
        dataLoader = DataLoader(graphs, batch_size=128)

    # Model Initialization
    model = Graph2Graph(graph_size=10, hidden_dim=124)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)

    plot_counter = 0
    for batch in tqdm.tqdm(dataLoader):
        total_loss = total_examples = prev_loss = 0
        for epoch in range(1, 100000):
            optimizer.zero_grad()

            X = torch.concat((batch.x, batch.coordinates), dim=1).to(torch.float32).to(device)
            pred = model.forward(X, batch.edge_index)

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
                    print(loss)
                    fig, axs = sample_draw_probs_graph(batch, pred)
                    fig.savefig(f"{plot_counter}.png")
                    plot_counter += 1
                    break
