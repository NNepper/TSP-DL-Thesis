# Libs
import torch.nn as nn


class PolicyFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.graph_size = config["num_nodes"]
        self.layer_dim = config["layer_size"]
        self.layer_number = config["layer_number"]
        self.input_dim = (self.graph_size * self.graph_size) + self.graph_size + 1

        # Dense-Layer
        self.dense = nn.Sequential(
            nn.Linear(self.input_dim, self.layer_dim),
            nn.ReLU(),
            nn.Linear(self.layer_dim, self.layer_dim),
            nn.ReLU()
        )

        for _ in range(self.layer_number - 2):
            self.dense.append(nn.Linear(self.layer_dim, self.layer_dim))
            self.dense.append(nn.ReLU())

        self.dense.append(nn.Linear(self.layer_dim, self.graph_size))

        # Output softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, mask):
        """
        The forward function takes in a batch of input sequences and masks,
        and returns the probability distribution over all possible next Nodes.
        The mask is used to prevent attention from being applied to visited Nodes.


        :param self: Represent the instance of the class
        :param input: Pass the input to the model
        :param mask: Mask the visited Nodes in the input
        :return: The probability distribution over the next Nodes
        """
        output = self.dense(input)
        output_masked = output.masked_fill(mask.bool(), float('-1e8'))
        probs = self.softmax(output_masked)
        return probs
