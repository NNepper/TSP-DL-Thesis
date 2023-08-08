import sys
import argparse
import os
import random
import csv
import math

import torch
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt

from common.loss import cross_entropy, cross_entropy_negative_sampling, cross_entropy_full
from common.scheduler import CosineWarmup
from common.visualization import draw_solution_graph
from data.dataset import TSPDataset
from model.model import Graph2Seq

# Argument
parser = argparse.ArgumentParser(description='TSP Solver using Supervised Graph2Seq model')
parser.add_argument('--data_train', type=str, default='tsp20_train_small.txt', help='Path to training dataset')
parser.add_argument('--data_test', type=str, default='data/tsp20_test.txt', help='Path to validation dataset')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size for training (default: 64)')
parser.add_argument('--num_nodes', type=int, default=20, help='number fo nodes in the graphs (default: 20)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--emb_dim', type=int, default=512, help='Size of the embedding vector (default: 128)')
parser.add_argument('--enc_hid_dim', type=int, default=2048, help='number of unit per dense layer in the Node-Wise Feed-Forward Network (default: 2048))')
parser.add_argument('--enc_num_layers', type=int, default=6, help='number of layer')
parser.add_argument('--enc_num_heads', type=int, default=8, help='number of Attention heads on Encoder')
parser.add_argument('--dec_num_heads', type=int, default=8, help='number of Attention heads on Decoder')
parser.add_argument('--drop_rate', type=float, default=.1, help='Dropout rate (default: .1)')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning multiplier for the NOAM schedule')      
parser.add_argument('--directory', type=str, default="./results", help='path where model and plots will be saved')
parser.add_argument('--n_gpu', type=int, default=0, help='number of GPUs to use (default: 2)')
parser.add_argument('--loss', type=str, default='full', help='loss function to use (default: negative_sampling)')
parser.add_argument('--teacher_forcing_constant', type=float, default=2.0, help='teacher forcing constant, larger increase teacher forcing in the schedule (default: 0.0, no teacher forcing)')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint to load weights from (default: None)')
parser.add_argument('--log', type=bool, default=True, help='Log the training (default: False)')
parser.add_argument('--norm_layer', type=str, default='batch', help='Normalization layer to use (Batch, Layer)')
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

with open(config.directory + "/metrics_validation.csv", 'w', newline='') as csv_val:
    writer_val = csv.writer(csv_val)
    writer_val.writerow(["Epoch", "Train_loss", "Test_loss", "Learning_rate"])

with open(config.directory + "/metrics_gradient.csv", 'w', newline='') as csv_grad:
    writer_grad = csv.writer(csv_grad)
    writer_grad.writerow(["Epoch", "Step", "Train_loss", "Gradient_norm", "Gradient_mean", "Gradient_std", "Gradient_max", "Gradient_min"])

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
        normalization=config.norm_layer
    ).to(device)

    # Data importing
    train_dataset = TSPDataset(config.data_train, config.num_nodes)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.n_gpu)
    test_dataset = TSPDataset(config.data_test, config.num_nodes)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, pin_memory=True, num_workers=0)

    # Multi-GPU sup'port
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)  # Wrap the model with DataParallel
        print(f"Using {torch.cuda.device_count()} GPUs !")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Load checkpoint if specified
    if config.checkpoint != None and os.path.exists(config.checkpoint):
        checkpoint = torch.load(config.checkpoint)
        epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        print(f"Resuming training from epoch {epoch}...")
    else:
        # start training from epoch 0
        epoch = 0
        print("Starting training from scratch...")

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
    grad_norm = torch.zeros(len(train_dataloader), len(list(model.parameters())))
    for epoch in range(epoch, config.epochs):
        model.train()
        train_loss = 0
        for i, (graph, target) in enumerate(train_dataloader):
            optimizer.zero_grad()

            graph = graph.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            probs, tours, loss = model(graph, target, teacher_forcing_constant=config.teacher_forcing_constant, loss_criterion=criterion)
            
            loss = loss.mean()
            loss.backward()
            train_loss += loss.item()

            # Gradient norm
            if config.log: 
                grad_norm[i,:] = torch.stack([torch.norm(p.grad.detach()) for p in model.parameters()])
                with open(config.directory + "/metrics_gradient.csv", 'a', newline='') as csv_grad:
                    writer_grad = csv.writer(csv_grad)
                    writer_grad.writerow([epoch+1, i+1, loss.item(), grad_norm[i].mean().item(), grad_norm[i].std().item(), grad_norm[i].max().item(), grad_norm[i].min().item()])

            optimizer.step()     # Update parameters

        train_loss /= len(train_dataloader)

        # Validation
        model.eval()
        test_loss = 0
        selected_plot = random.randrange(len(test_dataset))
        with torch.no_grad():
            graph, target = next(iter(test_dataloader))
            
            graph = graph.to(device)
            target = target.to(device)

            probs, outputs, loss = model(graph, target, teacher_forcing_constant=None, loss_criterion=criterion)

            test_loss = loss.mean()

            fig = draw_solution_graph(
                graph[selected_plot].cpu().detach().numpy(), 
                target[selected_plot].cpu().detach().numpy(), 
                outputs[selected_plot].cpu().detach().numpy())
            fig.savefig(
                    config.directory + "/G2S_" + str(config.num_nodes) + "_plot" + str(epoch) + ".png")
            plt.close(fig)
        # report metrics
        print(f"Epoch:{epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {test_loss:.6f}, grad_norm_std, {grad_norm.std():.10f}, grad_norm_mean, {grad_norm.mean():.10f}, grad_norm_range: [{grad_norm.min():.10f},{grad_norm.max():.10f}], Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        with open(config.directory + "/metrics_validation.csv", 'w', newline='') as csv_val:
            writer_val = csv.writer(csv_val)
            writer_val.writerow([epoch+1, train_loss, test_loss, optimizer.param_groups[0]["lr"]])

        # Save checkpoint
        checkpoint = {
                'epoch' : epoch,
                'model' : model.state_dict(),
                }
        torch.save(checkpoint, config.directory + "/checkpoint.pt")