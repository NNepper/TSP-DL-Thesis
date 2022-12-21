# Libs
import torch.nn as nn


class PolicyFeedForward(nn.Module):
    def __init__(self, graph_size, layer_dim, layer_number):
        super().__init__()
        input_dim = (graph_size * graph_size) + graph_size + 1

        # Dense-Layer
        self.dense = nn.Sequential(
            nn.Linear(input_dim, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, layer_dim),
            nn.ReLU()
        )

        for _ in range(layer_number - 2):
            self.dense.append(nn.Linear(layer_dim, layer_dim))
            self.dense.append(nn.ReLU())

        self.dense.append(nn.Linear(layer_dim, graph_size))

        # Output softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, mask):
        output = self.dense(input)
        output_masked = output.masked_fill(mask.bool(), float('-1e8'))
        probs = self.softmax(output_masked)
        return probs
