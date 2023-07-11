import argparse
import os
import pickle
import random

import torch
import tqdm
import math
import numpy as np
from torch_geometric.loader import DataLoader

from common.loss import cross_entropy, cross_entropy_negative_sampling
from common.visualization import sample_draw_probs_graph, draw_solution_graph
from model.model import Graph2Seq

# Argument
parser = argparse.ArgumentParser(description='TSP Solver using Supervised Graph2Seq model')
parser.add_argument('--batch-size', type=int, default=512, help='input batch size for training (default: 64)')
parser.add_argument('--num_nodes', type=int, default=20, help='number fo nodes in the graphs (default: 20)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--emb_dim', type=int, default=128, help='Size of the embedding vector (default: 128)')
parser.add_argument('--enc_hid_dim', type=int, default=512, help='number of unit per dense layer in the Node-Wise Feed-Forward Network (default: 512))')
parser.add_argument('--enc_num_layers', type=int, default=4, help='number of layer')
parser.add_argument('--enc_num_heads', type=int, default=4, help='number of Attention heads on Encoder')
parser.add_argument('--dec_num_layers', type=int, default=6, help='number of layer')
parser.add_argument('--dec_num_heads', type=int, default=4, help='number of Attention heads on Decoder')
parser.add_argument('--lr', type=float, default=.0001, help='learning rate')
parser.add_argument('--directory', type=str, default="./results_vanilla", help='path where model and plots will be saved')

config = parser.parse_args()
config.tuning = False

if __name__ == '__main__':
    # Data importing
    with open(f'data/dataset_{config.num_nodes}_train.pkl', 'rb') as f:
        graphs, target, opt_length = pickle.load(f)
        dataLoader = DataLoader(graphs, batch_size=config.batch_size)

    # Model Initialization
    model = Graph2Seq(
        dec_emb_dim=config.emb_dim,
        dec_num_layers=config.dec_num_layers,
        dec_num_heads=config.dec_num_heads,
        enc_emb_dim=config.emb_dim,
        enc_hid_dim=config.enc_hid_dim,
        enc_num_layers=config.enc_num_layers,
        enc_num_head=config.enc_num_heads,
        graph_size=config.num_nodes,
    )
    if os.path.exists(f'{config.directory}/G2S_{config.num_nodes}'):
        model.load_state_dict(
            torch.load(f'{config.directory}/G2S_{config.num_nodes}'))
        print("model loaded !")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    plot_counter = 0
    tours = []
    model.train()
    for epoch in range(1, 100):
        total_loss = total_examples = prev_loss = 0
        optimizer.zero_grad()
        for i, batch in tqdm.tqdm(enumerate(dataLoader), total=len(dataLoader)):
            x_batch = batch.x.float()
            edge_attr = batch.edge_attr.float()
            edge_index = batch.edge_index

            # Forward
            probs, tour = model.forward(x_batch, edge_index, edge_attr)
            loss = cross_entropy(probs, batch.y).sum() / config.batch_size

            # Backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            loss.backward()
            optimizer.step()

            # Report
            total_loss += loss.detach().cpu()
            for j in range(batch.num_graphs):
                tours.append(tour[j, :].detach().cpu().numpy())
        
        # Visualization
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / (math.ceil(len(graphs) / config.batch_size)):.4f}")
        selected = random.randrange(len(graphs))
        fig = draw_solution_graph(graphs[selected], tours[selected])
        fig.savefig(
            f'{config.directory}/G2S_{config.num_nodes}_plot{plot_counter}.png')
        plot_counter += 1

        # Model checkpoint
        torch.save(model.state_dict(),
                   f'{config.directory}/G2S_{config.num_nodes}')
