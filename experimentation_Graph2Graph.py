import argparse
import os
import pickle
import random

import torch
import tqdm
from torch_geometric.loader import DataLoader

from common.visualization import sample_draw_probs_graph, draw_probs_graph
from model.Graph2Graph import Graph2Graph


from common.loss import cross_entropy_negative_sampling, cross_entropy


# Argument
parser = argparse.ArgumentParser(description='TSP Solver using Supervised GNN model')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 1)')
parser.add_argument('--num_nodes', type=int, default=10, help='number fo nodes in the graphs (default: 10)')
parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train (default: 100)')
parser.add_argument('--layer_size', type=int, default=256, help='number of unit per dense layer')
parser.add_argument('--layer_number', type=int, default=3, help='number of layer')
parser.add_argument('--heads', type=int, default=1, help='number of Attention heads')
parser.add_argument('--lr', type=float, default=.0001, help='learning rate')
parser.add_argument('--directory', type=str, default="./results", help='path where model and plots will be saved')

config = parser.parse_args()
config.tuning = False

if __name__ == '__main__':
    # Data importing
    with open(f'data/dataset_{config.num_nodes}_train.pkl', 'rb') as f:
        graphs, target, opt_length = pickle.load(f)
        graphs = [graphs[0]]
        dataLoader = DataLoader(graphs, batch_size=1)

    # Model Initialization
    model = Graph2Graph(graph_size=config.num_nodes, hidden_dim=config.layer_size, num_layers=config.layer_number)
    # if os.path.exists(f'{config.directory}/GAT_{config.num_nodes}_{config.layer_size}_{config.layer_number}'):
    #     model.load_state_dict(
    #         torch.load(f'{config.directory}/GAT_{config.num_nodes}_{config.layer_size}_{config.layer_number}'))
    #     print("model loaded !")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    plot_counter = 0
    for epoch in range(1, config.epochs):
        for i, batch in enumerate(tqdm.tqdm(dataLoader, disable=True)):
            optimizer.zero_grad()

            X = batch.x.to(torch.float32)
            edge_attr = batch.edge_attr.to(torch.float32)
            edge_index = batch.edge_index

            pi = model.forward(X, edge_index, edge_attr)

            loss = cross_entropy_negative_sampling(pi, batch.y, 3).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2.0)

            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
            fig, axs = sample_draw_probs_graph(batch, pi)
            fig.savefig(
                f'{config.directory}/GAT_{config.num_nodes}_{config.layer_size}_{config.layer_number}_plot{plot_counter}.png')
            plot_counter += 1
            torch.save(model.state_dict(),
                    f'{config.directory}/GAT_{config.num_nodes}_{config.layer_size}_{config.layer_number}')
