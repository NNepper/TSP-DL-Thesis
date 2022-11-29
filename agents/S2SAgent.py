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
        """
        The forward function takes the input, feeds it through each of the layers in
        the model, and then returns the output and hidden state.


        :param self: Access the attributes and methods of the class
        :param input: Pass the input tensor through the rnn
        :return: The output and hidden state of the rnn
        """
        output, hidden = self.rnn(input)  # output > [num_layers], hidden > [hid_dim]
        return output, hidden


class Decoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, mask):  # input > [1, input_dim=2]
        """
        The forward function takes the input and hidden states, and runs the RNN model on them.
        It returns the output of that model, as well as its hidden state.

        :param self: Access variables that belongs to the class
        :param input: Pass the input data to the rnn
        :param hidden: Store the hidden state of the rnn
        :param mask: Mask the padded tokens
        :return: The output and the hidden state
        """
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
        """
        The __init__ function is the constructor for a class. It is called when you create an instance of a class.
        The __init__ method can have required arguments, optional arguments and keyword-only arguments (Python 3 only).
        In our case, we are using it to set up the model's architecture.

        :param self: Access variables that belongs to the class
        :param input_dim:int: Specify the size of the input dimension
        :param hidden_dim:int: Define the size of the hidden state
        :param output_dim:int: Specify the size of the output dimension
        :param enc_layers_size:int: Specify the number of layers in the encoder
        :param dec_layers_size:int: Specify the number of layers in the decoder
        :param csv_path:str=&quot;loss_log.csv&quot;: Specify the path to the csv file where we will log our loss values
        :param seed:int=69: Set the seed for the random number generator
        :return: The following:
        """
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
        self.csv_path = "loss_log.csv",

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


class S2SAgent:
    def __init__(self,
                 input_dim=2,
                 hidden_dim: int = 5,
                 output_dim: int = 10,
                 enc_layer_size: int = 2,
                 dec_layer_size: int = 2,
                 lr: float = 1e-4,
                 csv_path: str = "loss_log.csv",
                 seed=69):
        """
        The __init__ function is the constructor for a class. It is called when you create an instance of a class.
        The __init__ method can have required arguments, optional arguments and keyword-only arguments (Python 3 only).
        In our case, we are using it to set up the model's architecture.

        :param self: Access variables that belong to a class
        :param input_dim=2: Define the size of the input vector
        :param hidden_dim:int=5: Define the size of the hidden dimension
        :param output_dim:int=10: Define the size of the output vector
        :param enc_layer_size:int=2: Define the number of layers in the encoder
        :param dec_layer_size:int=2: Define the number of layers in the decoder
        :param lr:float=1e-4: Set the learning rate
        :param csv_path:str="loss_log.csv": Specify the path to a csv file where the loss will be stored
        :param seed=69: Ensure that the same training and validation sets are used for each model
        :return: The self object
        """
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
        """
        The train function trains the model for a given number of epochs.

        :param self: Access the class instance inside of the method
        :param env:TSPEnv: Pass the environment to the model
        :param epochs:int=100: Specify the number of epochs
        :return: The loss
        """
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