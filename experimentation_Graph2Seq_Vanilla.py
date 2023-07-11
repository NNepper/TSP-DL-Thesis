import argparse
import os
import random

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau;
from torch.utils.data import DataLoader

import math
import numpy as np

from common.loss import cross_entropy, cross_entropy_negative_sampling
from common.visualization import sample_draw_probs_graph, draw_solution_graph
from data.dataset import TSPDataset
from model.model import Graph2Seq

# Argument
parser = argparse.ArgumentParser(description='TSP Solver using Supervised Graph2Seq model')
parser.add_argument('--data', type=str, default='data/tsp20_ortools.txt', help='Path to data set')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 64)')
parser.add_argument('--num_nodes', type=int, default=20, help='number fo nodes in the graphs (default: 20)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--emb_dim', type=int, default=128, help='Size of the embedding vector (default: 128)')
parser.add_argument('--enc_hid_dim', type=int, default=512, help='number of unit per dense layer in the Node-Wise Feed-Forward Network (default: 512))')
parser.add_argument('--enc_num_layers', type=int, default=4, help='number of layer')
parser.add_argument('--enc_num_heads', type=int, default=4, help='number of Attention heads on Encoder')
parser.add_argument('--dec_num_layers', type=int, default=6, help='number of layer')
parser.add_argument('--dec_num_heads', type=int, default=4, help='number of Attention heads on Decoder')
parser.add_argument('--lr', type=float, default=.001, help='learning rate')
parser.add_argument('--directory', type=str, default="./results_van", help='path where model and plots will be saved')

config = parser.parse_args()
config.tuning = False

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name()} for training.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU for training.")

if __name__ == '__main__':
    # Data importing
    train_dataset = TSPDataset(
        filename='data/tsp20_ortools.txt',
        batch_size=config.batch_size,
        num_samples=128000,
        neighbors=-1,
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )

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
    ).to(device)
    if os.path.exists(f'{config.directory}/G2S_{config.num_nodes}'):
        model.load_state_dict(
            torch.load(f'{config.directory}/G2S_{config.num_nodes}'))
        print("model loaded !")

    # Optimizer and LR Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    plot_counter = 0
    tours = []
    model.train()
    for epoch in range(1, 1000):
        total_loss = total_examples = prev_loss = 0
        optimizer.zero_grad()
        for batch in train_dataloader:
            # Get batch
            x = batch["nodes"].float().to(device)

            # Forward
            probs, tour = model.forward(x)
            loss = cross_entropy(probs, batch["tour_nodes"]).sum() / config.batch_size

            # Backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            loss.backward()

            # Report
            total_loss += loss.detach().cpu().numpy()
            for j in range(config.batch_size):
                tours.append(tour[j, :].detach().cpu().numpy())    

            optimizer.step()

        # LR Scheduler
        scheduler.step(total_loss / (math.ceil(len(train_dataset) / config.batch_size)))
        
        # Visualization
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / (math.ceil(len(train_dataset) / config.batch_size)):.4f}, LR: {optimizer.param_groups[0]['lr']}")
        selected = random.randrange(len(train_dataset))
        fig = draw_solution_graph(train_dataset[selected], tours[selected])
        fig.savefig(
            f'{config.directory}/G2S_{config.num_nodes}_plot{plot_counter}.png')
        plot_counter += 1

        # Model checkpoint
        torch.save(model.state_dict(),
                   f'{config.directory}/G2S_{config.num_nodes}')
