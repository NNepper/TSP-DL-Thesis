import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt

from common.utils import compute_optimality_gap
from common.loss import cross_entropy, cross_entropy_negative_sampling, cross_entropy_full
from common.visualization import draw_solution_graph, draw_probability_graph
from data.dataset import TSPDataset
from model.model import Graph2Seq

# Argument
parser = argparse.ArgumentParser(description='TSP Solver using Supervised Graph2Seq model')
parser.add_argument('--data', type=str, default='data/tsp20_tiny.txt', help='Path to prediciton dataset')
parser.add_argument('--num_nodes', type=int, default=20, help='number fo nodes in the graphs (default: 20)')
parser.add_argument('--emb_dim', type=int, default=512, help='Size of the embedding vector (default: 128)')
parser.add_argument('--enc_hid_dim', type=int, default=4096, help='number of unit per dense layer in the Node-Wise Feed-Forward Network (default: 2048))')
parser.add_argument('--enc_num_layers', type=int, default=6, help='number of layer')
parser.add_argument('--enc_num_heads', type=int, default=8, help='number of Attention heads on Encoder')
parser.add_argument('--dec_num_heads', type=int, default=8, help='number of Attention heads on Decoder')
parser.add_argument('--drop_rate', type=float, default=.1, help='Dropout rate (default: .1)')
parser.add_argument('--directory', type=str, default="./results", help='path where model and plots will be saved')
parser.add_argument('--n_gpu', type=int, default=0, help='number of GPUs to use (default: 2)')
parser.add_argument('--loss', type=str, default='negative_sampling', help='loss function to use (default: negative_sampling)')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
parser.add_argument('--checkpoint', type=str, default="checkpoint.pt", help='Path to a checkpoint to load weights from (default: None)')
parser.add_argument('--log', type=bool, default=True, help='Log the training (default: False)')
parser.add_argument('--norm_layer', type=str, default='batch', help='Normalization layer to use (Batch, Layer)')
parser.add_argument('--aggregation', type=str, default='sum', help='Aggregation function to use (max, sum)')
config = parser.parse_args()
 
# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.device_count()} {torch.cuda.get_device_name()} for inference.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU for training.")

# Prepare metrics and results
if not os.path.exists(config.directory):
    os.makedirs(config.directory)

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
        normalization=config.norm_layer,
        aggregation=config.aggregation,
    ).to(device)

    # Data importing
    dataset = TSPDataset(config.data, config.num_nodes)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, pin_memory=True, num_workers=4)

    # Multi-GPU support
    model = nn.DataParallel(model)  # Wrap the model with DataParallel

    # Load checkpoint if specified
    if config.checkpoint != None and os.path.exists(config.checkpoint):
        checkpoint = torch.load(config.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
    else:
        raise FileNotFoundError

    # Loss function
    if config.loss == 'negative_sampling':
        criterion = cross_entropy_negative_sampling
    elif config.loss == 'full':
        criterion = cross_entropy_full
    elif config.loss == 'vanilla':
        criterion = cross_entropy
    else:
        raise NotImplementedError


    # Validation
    model.eval()
    test_loss = 0
    gaps = torch.zeros(len(dataset))
    with torch.no_grad():
        graph, target = next(iter(dataloader))
        
        graph = graph.to(device)
        target = target.to(device)

        probs, outputs, loss = model(graph, target, teacher_forcing_constant=None, loss_criterion=criterion)

        test_loss = loss.mean().item()

        for i in range(graph.shape[0]):
            g = graph[i].cpu().detach().numpy()
            t = target[i].cpu().detach().numpy()
            pred = outputs[i].cpu().detach().numpy()

            # Plot predicted
            fig1 = draw_solution_graph(g, t, pred)
            fig1.savefig(f"{config.directory}/G2S_{i}_plot.png")
            plt.close(fig1)

            # Plot probability assigned to a random nodes
            fig2 = draw_probability_graph(g, pred, probs)
            fig2.savefig(f"{config.directory}/G2S_{i}_probability_plot.png")

            gaps[i] = compute_optimality_gap(g, pred, t)
        
        # Display gaps statistics
        print(f"Mean:{gaps.mean():.3f}, std:{gaps.std():.3f}, range:[{gaps.min():.3f}:{gaps.max():.3f}]")