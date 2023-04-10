import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax


class DotDecoder(nn.Module):
    def __init__(self, graph_size):
        super().__init__()
        self.graph_size = graph_size

    def forward(self, x, edge_index):
        """
        The forward function takes in a batch of node features and returns the
        probability distribution over all possible edges. The probability is computed
        by taking the softmax of the dot product between each pair of nodes. This is
        equivalent to computing a similarity score for each pair, and then normalizing
        the scores so that they sum to 1.

        :param self: Access variables that belong to the class
        :param x: Pass the node features to the forward function
        :param edge_index: Construct the adjacency matrix
        :return: A tensor of size (batch_size, graph_size, graph_size)
        """
        pi = torch.zeros(x.shape[0] // self.graph_size, self.graph_size, self.graph_size)
        batch = torch.split(x, self.graph_size)
        edges = torch.split(edge_index, self.graph_size * (self.graph_size - 1), dim=1)
        for i, (x_batch, edge_idx_batch) in enumerate(zip(batch, edges)):
            logit = x_batch @ x_batch.t()

            # Compute softmax normalized for each node
            pi[i, :, :] = softmax(
                src=logit.view(self.graph_size * self.graph_size),
                index=torch.arange(0, self.graph_size).repeat(self.graph_size).to(torch.long)
            ).view(self.graph_size, self.graph_size)
        return pi


class RNNDecoder(nn.Module):
    def _init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
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
        output = F.Relu(output)
        output = self.out(output).masked_fill(mask.unsqueeze(1).bool(), float("-inf"))
        return output, hidden