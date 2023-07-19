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
parser.add_argument('--model_path', type=str, default="G2G", help="Path to the model's weights")
parser.add_argument('--emb_dim', type=int, default=512, help='Size of the embedding vector (default: 128)')
parser.add_argument('--enc_hid_dim', type=int, default=2048, help='number of unit per dense layer in the Node-Wise Feed-Forward Network (default: 2048))')
parser.add_argument('--enc_num_layers', type=int, default=6, help='number of layer')
parser.add_argument('--enc_num_heads', type=int, default=8, help='number of Attention heads on Encoder')
parser.add_argument('--dec_num_heads', type=int, default=8, help='number of Attention heads on Decoder')
parser.add_argument('--n_gpu', type=int, default=0, help='number of GPUs to use (default: 2)')
config = parser.parse_args()


if __name__ == '__main__':
    # DATA
    with open(config.data_path, 'rb') as f:
        graphs, target, opt_length = pickle.load(f)
        batch = list(DataLoader(graphs, batch_size=len(graphs)))[0]

    # MODEL
    model = Graph2Seq(
        dec_emb_dim=config.emb_dim,
        dec_num_heads=config.dec_num_heads,
        enc_emb_dim=config.emb_dim,
        enc_hid_dim=config.enc_hid_dim,
        enc_num_layers=config.enc_num_layers,
        enc_num_head=config.enc_num_heads,
        graph_size=config.num_nodes,
        drop_rate=0.0,
        teacher_forcing_ratio=0.0
    )    

    # TODO: Inference code here