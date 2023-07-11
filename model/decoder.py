import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from torch_geometric.utils import softmax

from model.layers import ScaledDotProductAttention

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
                index=torch.Tensor([[i] * self.graph_size for i in range(self.graph_size)]).view(-1).long()
            ).view(self.graph_size, self.graph_size)
        return pi

class MHADecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads=8, drop_rate=0.0):
        super().__init__()
        self.dropout = drop_rate
    
        self.linear_q = nn.Linear(3 * embedding_dim, embedding_dim)  # Query (Context embedding)
        self.linear_k = nn.Linear(embedding_dim, embedding_dim)      # Key (Nodes embedding)
        self.linear_v = nn.Linear(embedding_dim, embedding_dim)      # Value (Nodes embedding)
        self.linear_o = nn.Linear(embedding_dim, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.num_heads = num_heads

    def forward(self, context_emb, nodes_emb, mask=None):
        num_nodes = nodes_emb.shape[1]
        # input > [1, input_dim=2]
        q, k, v = self.linear_q(context_emb), self.linear_k(nodes_emb), self.linear_v(nodes_emb)

        q = q.unsqueeze(1)\
            .repeat(1, num_nodes, 1)\
            .reshape(q.shape[0], self.num_heads, num_nodes, q.shape[1] // self.num_heads)
        k = k.reshape(k.shape[0], self.num_heads, num_nodes, k.shape[2] // self.num_heads) # (batch_size, num_heads, graph_size, emb_per_heads)
        v = v.reshape(v.shape[0], self.num_heads, num_nodes, v.shape[2] // self.num_heads) # (batch_size, num_heads, graph_size, emb_per_heads)

        y = ScaledDotProductAttention()(q, k, v, mask)
        y = y.reshape(y.shape[0], y.shape[2], self.num_heads * y.shape[3])
        y = self.linear_o(y).squeeze()

        # Clipping within [-10, 10]
        y = 10 * self.tanh(y)

        # Masking 
        y = y.masked_fill(mask == 1, -1e9)

        # Softmax 
        y = self.softmax(y)      
        return y