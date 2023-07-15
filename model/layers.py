import torch.nn as nn
import torch.nn.functional as F

import math

class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2).repeat(1, key.shape[1], key.shape[2], 1)
            scores = scores.masked_fill(mask == 1, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class NodeWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=True)

        # initialize weights
        nn.init.uniform_(self.linear1.weight, a=0, b=1)
        nn.init.uniform_(self.linear2.weight, a=0, b=1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

