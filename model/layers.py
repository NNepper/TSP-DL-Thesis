import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from torch_geometric.utils import softmax

class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2).repeat(1, key.shape[1], key.shape[2], 1)
            scores = scores.masked_fill(mask == 1, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class NodeWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_rate=0.0):
        super().__init__()
        self.dropout = drop_rate
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        x = F.dropout(F.relu(self.linear1(x)), p=self.dropout, training=self.training)
        x = self.linear2(x)
        return x

