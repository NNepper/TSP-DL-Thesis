import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import MHAEncoder
from model.decoder import MHADecoder

class Graph2Seq(nn.Module):

    def __init__(self,
                 graph_size : int,
                 enc_hid_dim : int,
                 enc_emb_dim : int,
                 enc_num_layers: int,
                 enc_num_head: int,
                 dec_num_layers : int,
                 dec_emb_dim: int,
                 dec_num_heads: int
                 ):
        super().__init__()

        # Model
        self.graph_size = graph_size
        self.enc_num_layers = enc_num_layers
        self.enc_emb_dim = enc_emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.enc_num_heads = enc_num_head
        self.dec_num_layers = dec_num_layers
        self.dec_emb_dim = dec_emb_dim
        self.dec_num_heads = dec_num_heads
        self.encoder = MHAEncoder(embedding_dim=enc_emb_dim, ff_hidden_dim=enc_hid_dim, num_layers=self.enc_num_layers, num_heads=self.enc_num_heads)
        self.decoder = MHADecoder(embedding_dim=dec_emb_dim, num_heads=dec_num_heads)

        # Initial token
        self.token_1 = torch.zeros(enc_emb_dim)
        self.token_f = torch.zeros(enc_emb_dim)
        nn.init.uniform_(self.token_1, a=0, b=1)
        nn.init.uniform_(self.token_f, a=0, b=1)
        nn.Parameter(self.token_1)
        nn.Parameter(self.token_f)

    def forward(self, x):
        batch_size = x.shape[0]
        tours = torch.zeros(batch_size, self.graph_size)

        # Move initial token to same device as input
        self.token_1 = self.token_1.to(x.device)
        self.token_f = self.token_f.to(x.device)
        print(f"token_1: {self.token_1.device}")
        print(f"token_f: {self.token_f.device}")

        # Encoding the Graph
        nodes_emb = self.encoder.forward(x)
        print(f"nodes_emb: {nodes_emb.device}")

        # Computing Graph embedding
        graph_emb = nodes_emb.mean(dim=1)
        print(f"graph_emb: {nodes_emb.device}")

        # Decoder Inputs
        context_emb = torch.concat([
            graph_emb,
            self.token_1.repeat(batch_size).unsqueeze(0).view(batch_size, self.enc_emb_dim),
            self.token_f.repeat(batch_size).unsqueeze(0).view(batch_size, self.enc_emb_dim)
        ], dim=1)

        # Decoding the Tour
        start_emb = torch.zeros(batch_size, self.enc_emb_dim).to(x.device)
        print(f"start_emb: {start_emb.device}")
        probs = torch.zeros(batch_size, self.graph_size, self.graph_size).to(x.device)
        print(f"probs: {probs.device}")
        mask = torch.zeros(batch_size, self.graph_size).to(x.device)
        for i in range(self.graph_size):
            output = self.decoder.forward(context_emb=context_emb, nodes_emb=nodes_emb, mask=mask)
            print(f"output_dec: {output.device}")

            # Find probabilities
            prob = F.softmax(output, dim=1)
            dec_idx = prob.argmax(1)
            print("dec_idx:", dec_idx.device)

            # Prepare input for next timestep
            for j in range(batch_size):
                mask[j, dec_idx[j]] = 1.
                tours[j, i] = dec_idx[j]
                probs[j, i, :] = prob[j, :]
                if (i == 0):
                    start_emb[j, :] = nodes_emb[j, dec_idx[j], :]

            context_emb = torch.concat([
                graph_emb,
                start_emb,
                nodes_emb[torch.arange(batch_size), dec_idx, :]
            ], dim=1)

        return probs, tours