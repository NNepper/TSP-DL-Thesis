import argparse
import os
import random
import csv

import torch
from torch.optim.lr_scheduler import ExponentialLR;
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt

from common.loss import cross_entropy, cross_entropy_negative_sampling, cross_entropy_full
from common.visualization import draw_solution_graph
from data.dataset import TSPDataset
from model.model import Graph2Seq

# Argument
parser = argparse.ArgumentParser(description='TSP Solver using Supervised Graph2Seq model')
parser.add_argument('--data_train', type=str, default='data/tsp20_test.txt', help='Path to training dataset')
parser.add_argument('--data_test', type=str, default='data/tsp20_test.txt', help='Path to validation dataset')
parser.add_argument('--batch_size', type=int, default=20, help='input batch size for training (default: 64)')
parser.add_argument('--num_nodes', type=int, default=20, help='number fo nodes in the graphs (default: 20)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--emb_dim', type=int, default=512, help='Size of the embedding vector (default: 128)')
parser.add_argument('--enc_hid_dim', type=int, default=2048, help='number of unit per dense layer in the Node-Wise Feed-Forward Network (default: 2048))')
parser.add_argument('--enc_num_layers', type=int, default=6, help='number of layer')
parser.add_argument('--enc_num_heads', type=int, default=8, help='number of Attention heads on Encoder')
parser.add_argument('--dec_num_heads', type=int, default=8, help='number of Attention heads on Decoder')
parser.add_argument('--drop_rate', type=float, default=.1, help='Dropout rate (default: .1)')
parser.add_argument('--lr', type=float, default=.001, help='learning rate')
parser.add_argument('--directory', type=str, default="./results", help='path where model and plots will be saved')
parser.add_argument('--n_gpu', type=int, default=0, help='number of GPUs to use (default: 2)')
parser.add_argument('--loss', type=str, default='full', help='loss function to use (default: negative_sampling)')
parser.add_argument('--teacher_forcing', type=float, default=.5, help='teacher forcing ratio (default: .5)')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
parser.add_argument('--warmup_steps', type=int, default=10000, help='Number of warmup steps before reducing the learning rate in the scheduler')

config = parser.parse_args()

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name()} for training.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU for training.")

# Prepare metrics and results
if not os.path.exists(config.directory):
    os.makedirs(config.directory)

with open(config.directory + "/metrics.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Epoch", "train_loss", "test_loss", "mean_grad_norm", "learning_rate"])

if __name__ == '__main__':
    # Model definition
    model = Graph2Seq(
        dec_emb_dim=config.emb_dim,
        dec_num_heads=config.dec_num_heads,
        enc_emb_dim=config.emb_dim,
        enc_hid_dim=config.enc_hid_dim,
        enc_num_layers=config.enc_num_layers,
        enc_num_head=config.enc_num_heads,
        graph_size=config.num_nodes,
        drop_rate=config.drop_rate,
    ).to(device)

    # Data importing
    train_dataset = TSPDataset(config.data_train, config.num_nodes)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=False, num_workers=config.n_gpu)
    test_dataset = TSPDataset(config.data_test, config.num_nodes)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, pin_memory=False, num_workers=0)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)  # Wrap the model with DataParallel

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    lambda_decay = lambda epoch : 1 / (max(config.warmup_steps, epoch)**0.5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_decay)
    # Loss function
    if config.loss == 'negative_sampling':
        criterion = cross_entropy_negative_sampling
    elif config.loss == 'full':
        criterion = cross_entropy_full
    elif config.loss == 'vanilla':
        criterion = cross_entropy
    else:
        raise NotImplementedError

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        grad_norm = torch.zeros(len(train_dataloader))
        for i, (graph, target) in enumerate(train_dataloader):
            optimizer.zero_grad() 
            
            graph = graph.to(device)
            target = target.to(device)
            
            probs, outputs = model(graph)
            loss = criterion(probs, target).mean()
            train_loss += loss.item()

            loss.backward()
            grad_norm[i] = torch.nn.utils.clip_grad_norm_(model.parameters(), 5).item()       
            optimizer.step()     # Apply the weight update
            scheduler.step()     # Update the learning rate following schedule

        train_loss /= len(train_dataloader)

        # Validation
        model.eval()
        tours = []
        test_loss = 0
        selected_plot = random.randrange(len(test_dataset))
        with torch.no_grad():
            graph, target = next(iter(test_dataloader))
            graph = graph.to(device)
            target = target.to(device)

            probs, tour = model(graph, target, teacher_forcing_ratio=0.0)

            loss = criterion(probs, target).mean()
            test_loss += loss.item()

            for j in range(len(outputs)):
                tours.append(outputs[j].cpu().numpy())

            # Plot the selected test graph
            if i == selected_plot:
                fig = draw_solution_graph(graph.squeeze().detach().cpu().numpy(), target.squeeze().detach().cpu().numpy(), tour.squeeze().detach().cpu().numpy())
                fig.savefig(
                        config.directory + "/G2S_" + str(config.num_nodes) + "_plot" + str(epoch) + ".png"
                        )
        # report metrics
        print("Epoch:", str(epoch+1), "Train Loss:", np.round(train_loss, 3), "Val Loss:", np.round(test_loss, 3), "Mean Grad Norm:", np.round(grad_norm.mean().item(), 3), "Learning rate:", scheduler.get_last_lr()[0])
        with open(config.directory + "/metrics.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch+1, train_loss, test_loss, grad_norm.mean(), scheduler.get_last_lr()])

        # Save model
        torch.save(model.state_dict(), config.directory + "/model.pt")

