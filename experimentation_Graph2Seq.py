import argparse
import os
import random

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau;
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler

import math
import numpy as np

from common.loss import cross_entropy, cross_entropy_negative_sampling, cross_entropy_full
from common.visualization import sample_draw_probs_graph, draw_solution_graph
from data.dataset import TSPDataset
from model.model import Graph2Seq

# Argument
parser = argparse.ArgumentParser(description='TSP Solver using Supervised Graph2Seq model')
parser.add_argument('--data_train', type=str, default='data/tsp20_train.txt', help='Path to training dataset')
parser.add_argument('--data_test', type=str, default='data/tsp20_val.txt', help='Path to validation dataset')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
parser.add_argument('--num_nodes', type=int, default=20, help='number fo nodes in the graphs (default: 20)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--emb_dim', type=int, default=128, help='Size of the embedding vector (default: 128)')
parser.add_argument('--enc_hid_dim', type=int, default=512, help='number of unit per dense layer in the Node-Wise Feed-Forward Network (default: 512))')
parser.add_argument('--enc_num_layers', type=int, default=4, help='number of layer')
parser.add_argument('--enc_num_heads', type=int, default=4, help='number of Attention heads on Encoder')
parser.add_argument('--dec_num_layers', type=int, default=6, help='number of layer')
parser.add_argument('--dec_num_heads', type=int, default=4, help='number of Attention heads on Decoder')
parser.add_argument('--drop_rate', type=float, default=.1, help='Dropout rate (default: .1)')
parser.add_argument('--lr', type=float, default=.001, help='learning rate')
parser.add_argument('--directory', type=str, default="./results", help='path where model and plots will be saved')
parser.add_argument('--n_gpu', type=int, default=2, help='number of GPUs to use (default: 2)')
parser.add_argument('--loss', type=str, default='negative_sampling', help='loss function to use (default: negative_sampling)')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

config = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # Model definition
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
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Loss function
    if config.loss == 'negative_sampling':
        criterion = cross_entropy_negative_sampling
    elif config.loss == 'full':
        criterion = cross_entropy_full
    elif config.loss == 'vanilla':
        criterion = cross_entropy
    else:
        raise NotImplementedError(f"Loss function {config.loss} not implemented.")

    # Data importing
    train_dataset = TSPDataset(config.data_train, config.num_nodes)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.n_gpu)
    test_dataset = TSPDataset(config.data_test, config.num_nodes)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.n_gpu)

    # Training loop
    scaler = GradScaler()
    for epoch in range(config.epochs):
        model.train()
        test_loss = train_loss = 0
        tours = []
        for i, (graph, solution) in enumerate(train_dataloader):
            graph = graph.cuda()
            solution = solution.cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                probs, outputs = model(graph)
                loss = criterion(probs, solution).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation
        model.eval()
        with torch.no_grad():
            for i, (graph, solution) in enumerate(test_dataloader):
                probs, tour = model(graph)

                loss = criterion(probs, solution).mean()
                test_loss += loss.item()

                for j in range(len(outputs)):
                    tours.append(outputs[j].cpu().numpy())

            test_loss /= len(test_dataloader)
            scheduler.step(test_loss)
            test_loss = test_loss / len(test_dataloader)

            scheduler.step(test_loss)

        # Save model
        if (epoch + 1) % 10 == 0:
            torch.save(model.module.state_dict(), os.path.join(config.directory, f"model_{epoch + 1}.pt"))

            # Plot solution
            selected = random.randrange(len(train_dataset))
            fig = draw_solution_graph(train_dataset[selected], tours[selected])
            fig.savefig(
                f'{config.directory}/G2S_{config.num_nodes}_plot{epoch}.png')

        # Print statistics
        print(f"Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Val Loss: {test_loss:.4f}")
