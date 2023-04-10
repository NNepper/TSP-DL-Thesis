import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import GNNEncoder
from model.decoder import RNNDecoder

class Graph2Seq(nn.Module):

    def __init__(self,
                 graph_size : int,
                 enc_num_layers: int,
                 enc_hid_dim : int,
                 dec_num_layers : int,
                 dec_hid_dim: int,
                 ):
        super().__init__()
        # Model
        self.enc_num_layers = enc_num_layers
        self.enc_hid_dim = enc_hid_dim
        self.dec_num_layers = dec_num_layers
        self.dec_hid_dim = dec_hid_dim
        self.encoder = GNNEncoder(input_dim=2, hidden_dim=enc_hid_dim, num_layers=enc_num_layers)
        self.decoder = RNNDecoder(input_dim=enc_hid_dim, hidden_dim=dec_hid_dim, num_layers=dec_num_layers)

        # Pytorch Env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x, edge_index, edge_attributes):
        tour = [[] for _ in range(x.shape[0])]
        mask = torch.zeros((x.shape[0], self.graph_size))

        # Encoding the Graph
        _, hidden = self.encoder.forward(x, edge_index, edge_attributes)

        # Decoding the Tour
        # dec_in = torch.zeros((x.shape[0], self.input_dim), device=self.device)
        acc_loss = torch.zeros((x.shape[0],), device=self.device)

        while not done:
            output, _ = self.decoder.forward(dec_in, hidden, mask=mask)

            # Find probabilities
            prob = F.softmax(output, dim=2)
            dec_idx = prob.argmax(2)

            # Compute loss
            _, loss, done, _ = env.step(dec_idx)

            # Store result
            state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)
            acc_loss += torch.tensor(loss, dtype=torch.float, device=self.device)

            # Prepare input for next timestep
            dec_in = torch.squeeze(torch.squeeze(state[:, dec_idx, :2], 1), 1)  # TODO: Modify the squeezing
        return acc_loss