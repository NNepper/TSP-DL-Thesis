# Libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

from gym_vrp.envs import TSPEnv


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=False):
        super().__init__()
        self.rnn = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional
        )

    def forward(self, input):  # input  > [src_len, input_dim=2]
        output, hidden = self.rnn(input)  # output > [num_layers], hidden > [hid_dim]
        return output, hidden


class Decoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, mask):  # input > [1, input_dim=2]
        output, hidden = self.rnn(input, hidden)
        output = torch.tanh(output)
        output = self.out(output).masked_fill(mask.unsqueeze(1).bool(), float("-inf"))
        return output, hidden


class Seq2Seq(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 enc_layers_size: int,
                 dec_layers_size: int,
                 csv_path: str = "loss_log.csv",
                 seed: int = 69):
        super().__init__()
        # Model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.enc_layers_size = enc_layers_size
        self.dec_layers_size = dec_layers_size
        self.encoder = Encoder(input_dim, hidden_dim, num_layers=enc_layers_size)
        self.decoder = Decoder(input_dim, hidden_dim, output_dim=output_dim, num_layers=dec_layers_size)

        # Output result
        self.csv_path: str = "loss_log.csv",

        # Pytorch Env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, env: TSPEnv):
        state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)

        # Encoding the Graph
        _, hidden = self.encoder(torch.squeeze(state[:, :, :2], 0))

        # Decoding the Tour
        dec_in = torch.zeros((state.shape[0], self.input_dim), device=self.device)
        done = False
        acc_loss = torch.zeros((state.shape[0],), device=self.device)

        while not done:
            mask = state[:, :, 3]
            output, _ = self.decoder(dec_in, hidden, mask=mask)

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


class BasicAgent:
    def __init__(self,
                 input_dim=2,
                 hidden_dim: int = 5,
                 output_dim: int = 10,
                 enc_layer_size: int = 2,
                 dec_layer_size: int = 2,
                 lr: float = 1e-4,
                 csv_path: str = "loss_log.csv",
                 seed=69):
        # Torch configuration
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        # Model
        self.model = Seq2Seq(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            enc_layers_size=enc_layer_size,
            dec_layers_size=dec_layer_size,
        ).to(self.device)

        # Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, env: TSPEnv, epochs: int = 100):
        for i in range(epochs):
            self.model.train()
            env.reset()

            # Predict tour
            loss = self.model(env)

            # back-prop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # report
            print(f"epoch {i}: loss={loss}")