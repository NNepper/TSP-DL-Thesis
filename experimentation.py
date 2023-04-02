import argparse
import pickle

import torch
import tqdm
from torch_geometric.loader import DataLoader

from common.loss import cross_entropy_negative_sampling
from common.visualization import sample_draw_probs_graph
from model.Graph2Graph import Graph2Graph

# Argument
parser = argparse.ArgumentParser(description='TSP Solver using Supervised GNN model')
parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training (default: 1)')
parser.add_argument('--num_nodes', type=int, default=20, help='number fo nodes in the graphs (default: 10)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--layer_size', type=int, default=256, help='number of unit per dense layer')
parser.add_argument('--layer_number', type=int, default=6, help='number of layer')
parser.add_argument('--heads', type=int, default=4, help='number of Attention heads')
parser.add_argument('--lr', type=float, default=.001, help='learning rate')
parser.add_argument('--directory', type=str, default="./results", help='path where model and plots will be saved')

config = parser.parse_args()
config.tuning = False

if __name__ == '__main__':
    # Data importing
    with open(f'data/dataset_{config.num_nodes}_train.pkl', 'rb') as f:
        graphs, target, opt_length = pickle.load(f)
        dataLoader = DataLoader(graphs, batch_size=config.batch_size)

    # Model Initialization
    model = Graph2Graph(graph_size=config.num_nodes, hidden_dim=config.layer_size, num_layers=config.layer_number,
                        num_heads=config.heads)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    plot_counter = 0
    for epoch in range(1, 100):
        total_loss = total_examples = prev_loss = 0
        for batch in tqdm.tqdm(dataLoader):
            optimizer.zero_grad()

            X = batch.x.to(torch.float32)
            edge_attr = batch.edge_attr.to(torch.float32)
            edge_index = batch.edge_index

            pi = model.forward(X, edge_index, edge_attr)

            loss = cross_entropy_negative_sampling(pi, batch.y, 10).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += loss.sum()
            total_examples += pi.numel()
        if abs(total_loss - prev_loss) > 10e-6:
            prev_loss = total_loss
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
        fig, axs = sample_draw_probs_graph(batch, pi)
        fig.savefig(
            f'{config.directory}/GAT_{config.num_nodes}_{config.layer_size}_{config.layer_number}_plot{plot_counter}.png')
        plot_counter += 1
        torch.save(model.state_dict,
                   f'{config.directory}/GAT_{config.num_nodes}_{config.layer_size}_{config.layer_number}')
