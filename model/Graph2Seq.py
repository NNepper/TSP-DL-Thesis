import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from common.loss import cross_entropy_negative_sampling

from model.encoder import PointNetEncoder, GATEncoder
from model.decoder import MHADecoder

class Graph2Seq(nn.Module):

    def __init__(self,
                 graph_size : int,
                 enc_num_layers: int,
                 enc_hid_dim : int,
                 enc_num_head: int,
                 dec_num_layers : int,
                 dec_hid_dim: int,
                 dec_num_heads: int,
                 ):
        super().__init__()
        # Model
        self.graph_size = graph_size
        self.enc_num_layers = enc_num_layers
        self.enc_hid_dim = enc_hid_dim
        self.enc_num_heads = enc_num_head
        self.dec_num_layers = dec_num_layers
        self.dec_hid_dim = dec_hid_dim
        self.dec_num_heads = dec_num_heads
        self.encoder = GATEncoder(hidden_dim=enc_hid_dim, num_layers=enc_num_layers, num_heads=enc_num_head)
        self.decoder = MHADecoder(embedding_dim=enc_hid_dim, num_heads=dec_num_heads)

        # Initial token
        self.token_1 = torch.empty(enc_hid_dim)
        self.token_f = torch.empty(enc_hid_dim )
        nn.init.uniform(self.token_1, a=0, b=1)
        nn.init.uniform(self.token_f, a=0, b=1)
        nn.Parameter(self.token_1)
        nn.Parameter(self.token_f)

        # Pytorch Env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x, edge_index, edge_attributes):
        batch_size = math.ceil(x.shape[0] / self.graph_size)
        tours = torch.zeros(batch_size, self.graph_size)

        # Encoding the Graph
        nodes_emb = torch.cat(
            torch.chunk(self.encoder.forward(x, edge_index, edge_attributes).unsqueeze(0),
                        chunks=batch_size,
                        dim=1)
        )

        # Computing Graph embedding
        graph_emb = nodes_emb.mean(dim=1)

        # Decoder Inputs
        context_emb = torch.concat([
            graph_emb,
            self.token_1.repeat(batch_size).unsqueeze(0).view(batch_size, self.enc_hid_dim),
            self.token_f.repeat(batch_size).unsqueeze(0).view(batch_size, self.enc_hid_dim)
        ], dim=1)

        # Decoding the Tour
        start_emb = torch.cat([nodes_emb[i, int(start[i])].unsqueeze(0) for i in range(batch_size)])
        probs = torch.zeros(batch_size, self.graph_size, self.graph_size)
        mask = torch.zeros(batch_si>e, self.graph_size)
        for i in range(self.graph_size):
            output = self.decoder.forward(context_emb=context_emb, nodes_emb=nodes_emb, mask=mask)

            # Find probabilities
            prob = F.softmax(output, dim=1)
            dec_idx = prob.argmax(1)

            # Prepare input for next timestep
            for j in range(batch_size):
                mask[j, dec_idx[j]] = 1.
                tours[j, i] = dec_idx[j]
                probs[j, i, :] = prob[j, :]

            context_emb = torch.concat([
                graph_emb,
                start_emb,
                nodes_emb[torch.arange(batch_size), dec_idx, :]
            ], dim=1)

        return probs, tours
