import torch
import torch.nn as nn

from model.layers import ScaledDotProductAttention, NodeWiseFeedForward

class GATEncoder(nn.Module):
    def __init__(self, hidden_dim, drop_rate=0.0, num_layers=4, num_heads=4):
        super().__init__()
        
    def forward(self, x, edge_index, edge_attr):
        raise(NotImplementedError)

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8, drop_rate=0.0):
        super().__init__()
        self.dropout = drop_rate
    
        self.linear_q = nn.Linear(input_dim, input_dim)  # Query
        self.linear_k = nn.Linear(input_dim, input_dim)  # Key 
        self.linear_v = nn.Linear(input_dim, input_dim)  # Value

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.num_heads = num_heads

    def forward(self, x, mask=None):
        num_nodes = x.shape[1]
        # input > [1, input_dim=2]
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)

        q = q.reshape(q.shape[0], self.num_heads, num_nodes, q.shape[2] // self.num_heads) # (batch_size, num_heads, graph_size, emb_per_heads)
        k = k.reshape(k.shape[0], self.num_heads, num_nodes, k.shape[2] // self.num_heads) # (batch_size, num_heads, graph_size, emb_per_heads)
        v = v.reshape(v.shape[0], self.num_heads, num_nodes, v.shape[2] // self.num_heads) # (batch_size, num_heads, graph_size, emb_per_heads)

        y = ScaledDotProductAttention()(q, k, v, mask)
        y = y.reshape(y.shape[0], y.shape[2], self.num_heads * y.shape[3])

        return y


class MHAEncoder(nn.Module):
    def __init__(self, embedding_dim=128, ff_hidden_dim=512, drop_rate=0.0, num_layers=4, num_heads=4):
        super().__init__()
        self.dropout = drop_rate
        self.num_layers = num_layers

        # Initial embedding
        self.linear0 = nn.Linear(2, embedding_dim)

        # Batch Normalization
        self.bn = nn.BatchNorm1d(embedding_dim, affine=False)
        self.bn_w = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, embedding_dim)).squeeze())
        self.bn_b = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, embedding_dim)).squeeze())

        # Multi-Head Attention Layers
        self.mha_layers = nn.ModuleList([
            MultiHeadAttention(embedding_dim, num_heads, drop_rate) for _ in range(num_layers)
        ])

        # (Sub) Node-wise Feed Forward Layers
        self.ff_layers = nn.ModuleList([
            NodeWiseFeedForward(input_dim=embedding_dim, hidden_dim=ff_hidden_dim, output_dim=embedding_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        # Initial embedding
        h = self.linear0(x)

        for i in range(self.num_layers):
            # Multi-Head Attention
            h_mha = self.mha_layers[i](h)

            # Batch Normalization
            h = (h_mha + h).transpose(1,2)
            h = self.bn(h).transpose(1,2)
            h = h * self.bn_w + self.bn_b

            # Node-wise Feed Forward
            h_ff = self.ff_layers[i](h)

            # Batch Normalization
            h = (h_ff + h).transpose(1,2)
            h = self.bn(h).transpose(1,2)
            h = h * self.bn_w + self.bn_b

        return h