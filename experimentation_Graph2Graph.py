import argparse
import os
import random

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau;
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

import math
import numpy as np

from common.loss import cross_entropy, cross_entropy_negative_sampling, cross_entropy_full
from common.visualization import sample_draw_probs_graph, draw_solution_graph
from data.dataset import TSPDataset
from model.model import Graph2Seq
from common.utils import load_dataset

# Argument
parser = argparse.ArgumentParser(description='TSP Solver using Supervised Graph2Seq model')
parser.add_argument('--train_data', type=str, default='data/tsp20_val.txt', help='Path to training dataset')
parser.add_argument('--val_data', type=str, default='data/tsp20_val.txt', help='Path to validation dataset')
parser.add_argument('--batch-size', type=int, default=10, help='input batch size for training (default: 64)')
parser.add_argument('--num_nodes', type=int, default=20, help='number fo nodes in the graphs (default: 20)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--emb_dim', type=int, default=128, help='Size of the embedding vector (default: 128)')
parser.add_argument('--enc_hid_dim', type=int, default=512, help='number of unit per dense layer in the Node-Wise Feed-Forward Network (default: 512))')
parser.add_argument('--enc_num_layers', type=int, default=4, help='number of layer')
parser.add_argument('--enc_num_heads', type=int, default=4, help='number of Attention heads on Encoder')
parser.add_argument('--dec_num_layers', type=int, default=6, help='number of layer')
parser.add_argument('--dec_num_heads', type=int, default=4, help='number of Attention heads on Decoder')
parser.add_argument('--lr', type=float, default=.001, help='learning rate')
parser.add_argument('--directory', type=str, default="./results_ns", help='path where model and plots will be saved')
parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--loss', type=str, default='negative_sampling', help='loss function to use (default: negative_sampling, other: full, vanilla)')

config = parser.parse_args()

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name()} for training.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU for training.")

if __name__ == '__main__':
    # Data importing
    train_dataset, train_dataloader = load_dataset(
        filename=config.train_data,
        batch_size=config.batch_size,
        neighbors=-1,
        workers=config.n_gpu
    )
    val_dataset, val_dataloader  = load_dataset(
        filename=config.val_data,
        neighbors=-1,
        workers=config.n_gpu
    )

    # Model Definition
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
    model = torch.nn.DataParallel(model)  # Wrap the model with DataParallel

    # Load model weights if exists
    if os.path.exists(f'{config.directory}/G2S_{config.num_nodes}'):
        model.load_state_dict(
            torch.load(f'{config.directory}/G2S_{config.num_nodes}'))
        print("model loaded !")

    # Optimizer and LR Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Loss function
    if config.loss == 'full':
        criterion = cross_entropy_full
    elif config.loss == 'negative_sampling':
        criterion = cross_entropy_negative_sampling
    elif config.loss == 'vanilla':
        criterion = cross_entropy

    scaler = GradScaler()
    model.to(device)
    for epoch in range(1, config.epochs + 1):
        train_loss = val_loss = 0
        tours = []

        # Training loop
        model.train()
        for i, (graphs, solutions) in enumerate(train_dataloader):
            graph = graphs.to(device)
            solutions = solutions.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                probs, outputs = model(graphs)
                loss = cross_entropy_negative_sampling(probs, solutions, n_neg=5)
                loss = (loss.sum() / len(graphs))
                train_loss += loss.item()


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss /= len(train_dataloader)

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, (graphs, solutions) in enumerate(val_dataloader):
                graphs = graphs.to(device)
                solutions = solutions.to(device)

                probs, outputs = model(graph)
                loss = cross_entropy_negative_sampling(probs, solutions, n_neg=5)
                val_loss += (loss.sum() / len(graphs)).item()
                
                # Save tours
                for j in range(len(graphs)):
                    tours.append(outputs[j].detach().numpy())

            val_loss /= len(val_dataloader)
            scheduler.step(val_loss)

        print(f"Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            # Save model
            torch.save(model.module.state_dict(), os.path.join(config.directory, f"model_{epoch + 1}.pt"))

            # Plot solution
            selected = random.randrange(len(val_dataset))
            fig = draw_solution_graph(val_dataloader[selected], tours[selected])
            fig.savefig(
                f'{config.directory}/G2S_{config.num_nodes}_plot{epoch}.png')