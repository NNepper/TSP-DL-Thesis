# Libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim : int, layer_dim: int, layer_number: int):
        # Dense-Layer
        self.dense = nn.Sequential(
            nn.Linear(input_dim, layer_dim),
            nn.ReLU())
        for _ in range(layer_number - 2):
            self.dense.append(nn.Linear(layer_dim, layer_dim))
            self.dense.append(nn.ReLU())
        self.dense.append(nn.Linear(layer_dim, output_dim))

        # Output
        self.output = F.softmax(output_dim, output_dim)

    def forward(self):
        return ...


class BasicAgent:
    def __init__(self, graph_size: int, layer_dim: int, layer_number: int):
        input_dim = (graph_size * 2) + graph_size + 1
        self.mlp = MLP(
            input_dim=input_dim,
            layer_dim=layer_dim,
            layer_number=layer_number
        )



    def train(self):
        return ...
