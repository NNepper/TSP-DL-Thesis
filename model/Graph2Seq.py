import torch
import torch.nn as nn

from model.encoder import GNNEncoder
from model.decoder import RNNDecoder

class Graph2Seq(nn.Module):

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
        self.encoder = GNNEncoder(input_dim, hidden_dim, num_layers=enc_layers_size)
        self.decoder = RNNDecoder(input_dim, hidden_dim, output_dim=output_dim, num_layers=dec_layers_size)

        # Output result
        self.csv_path = "loss_log.csv",

        # Pytorch Env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, ):
        state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)

        # Encoding the Graph
        _, hidden = self.encoder.forward(torch.squeeze(state[:, :, :2], 0))

        # Decoding the Tour
        dec_in = torch.zeros((state.shape[0], self.input_dim), device=self.device)
        done = False
        acc_loss = torch.zeros((state.shape[0],), device=self.device)

        while not done:
            mask = state[:, :, 3]
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