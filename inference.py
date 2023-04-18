import pickle

import argparse
import torch
import tqdm
from torch_geometric.loader import DataLoader

from common.loss import *
from common.utils import *
from common.visualization import draw_solution_graph
from common.search import BFSearch

from model.Graph2Graph import Graph2Graph
from model.Graph2Seq import Graph2Seq

# Argument
parser = argparse.ArgumentParser(description='TSP Solver using Supervised GNN model')
parser.add_argument('--num_nodes', type=int, default=20, help='number of nodes in the graphs (default: 10)')
parser.add_argument('--data_path', type=str, help='Path to the dataset generated using the TSP_Dataset notebook')
parser.add_argument('--model_type', type=str, default="Graph2Graph", help='Neural-Net architecture to be used by the model (default: Graph2Graph)')
parser.add_argument('--model_path', type=str, default="G2G", help="Path to the model's weights")
parser.add_argument('--directory', type=str, default="./results", help='path where model and plots will be saved')
config = parser.parse_args()


if __name__ == '__main__':
    # DATA
    with open(config.data_path, 'rb') as f:
        graphs, target, opt_length = pickle.load(f)
        batch = list(DataLoader(graphs, batch_size=len(graphs)))[0]

    # MODEL
    if config.model_type == "Graph2Graph":
        model = Graph2Graph(graph_size=config.num_nodes, hidden_dim=200)
    else:
        model = Graph2Seq(graph_size=config.num_nodes, hidden_dim=200)
    model.load_state_dict(torch.load(config.model_path))

    # SEARCH
    search = BFSearch()

    # Forward pass of the test examples
    X = batch.x.to(torch.float32)
    edge_attr = batch.edge_attr.to(torch.float32)
    edge_index = batch.edge_index

    pi = model.forward(X, batch.edge_index)

    # Decode the tour given the probability Heatmap
    preds, log_probs = search.predict(pi, torch.zeros(pi.shape[0]).int())

    # Compute the optimality gap of each solution given the optimum
    loss = compute_optimality_gap(batch, preds)

    # Visualisation
    for i in range(batch.num_graphs):
        fig = draw_solution_graph(batch[i], preds[i])
        fig.savefig(f"{config.directory}/{i}.png")
