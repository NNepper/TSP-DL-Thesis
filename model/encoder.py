import torch
import torch.nn as nn

from model.layers import ScaledDotProductAttention, NodeWiseFeedForward

class GATEncoder(nn.Module):
    def __init__(self, hidden_dim, drop_rate=0.0, num_layers=4, num_heads=4):
        super().__init__()
        
    def forward(self, x, edge_index, edge_attr):
        raise(NotImplementedError)

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super().__init__()
    
        self.linear_q = nn.Linear(input_dim, input_dim) # Query
        self.linear_k = nn.Linear(input_dim, input_dim) # Key 
        self.linear_v = nn.Linear(input_dim, input_dim) # Value

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.num_heads = num_heads

        # Weight Initalization
        nn.init.uniform_(self.linear_q.weight, a=0, b=1)
        nn.init.uniform_(self.linear_k.weight, a=0, b=1)
        nn.init.uniform_(self.linear_v.weight, a=0, b=1)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        emb_dim = x.shape[2]

        # input > [1, input_dim=2]
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)

        q = q.reshape(batch_size, num_nodes, self.num_heads, emb_dim // self.num_heads) # (batch_size, num_heads, graph_size, emb_per_heads)
        k = k.reshape(batch_size, num_nodes, self.num_heads, emb_dim // self.num_heads) # (batch_size, num_heads, graph_size, emb_per_heads)
        v = v.reshape(batch_size, num_nodes, self.num_heads, emb_dim // self.num_heads) # (batch_size, num_heads, graph_size, emb_per_heads)

        y = ScaledDotProductAttention()(q, k, v, mask)
        y = y.reshape(batch_size, num_nodes, emb_dim)

        return y


class MHAEncoder(nn.Module):
    def __init__(self, embedding_dim=128, ff_hidden_dim=512, num_layers=4, num_heads=4, drop_rate=0.0, normalization="batch", aggregation="sum"):
        super().__init__()
        self.num_layers = num_layers

        # Initial embedding
        self.linear0 = nn.Linear(2, embedding_dim)

        # Multi-Head Attention Layers
        self.mha_layers = nn.ModuleList([
            MultiHeadAttention(embedding_dim, num_heads) for _ in range(num_layers)
        ])

        # (Sub) Node-wise Feed Forward Layers
        self.ff_layers = nn.ModuleList([
            NodeWiseFeedForward(input_dim=embedding_dim, hidden_dim=ff_hidden_dim, output_dim=embedding_dim) for _ in range(num_layers)
        ])

        # Normalization layers
        if (normalization == "batch"):
            self.norm_layers = nn.ModuleList([
                nn.BatchNorm1d(embedding_dim, affine=True) for _ in range(num_layers)
            ])
        elif (normalization == "layer"):
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(embedding_dim, elementwise_affine=True) for _ in range(num_layers)
            ])
        else:
            raise NotImplementedError

        # Dropout
        self.dropout = nn.Dropout(drop_rate)

        # Aggregation
        if aggregation == "sum":
            self.aggregation = lambda x, y : x + y
        elif aggregation == "max":
            self.aggregation = lambda x, y : torch.max(x, y)
        else:
            raise NotImplementedError

        # Weight Initalization
        nn.init.uniform_(self.linear0.weight, a=0, b=1)
        nn.init.uniform_(self.linear0.bias, a=0, b=1)
        for i in range(num_layers):
            nn.init.uniform_(self.norm_layers[i].weight, a=0, b=1)
            nn.init.uniform_(self.norm_layers[i].bias, a=0, b=1)

    def forward(self, x):
        # Initial embedding
        h = self.linear0(x)

        for i in range(self.num_layers):
            # Multi-Head Attention
            h_mha = self.mha_layers[i](h)
            h_mha = self.dropout(h_mha)

            # Normalization
            h = self.aggregation(h_mha, h)
            h = self.norm_layers[i](h.transpose(1,2)).transpose(1,2)

            # Node-wise Feed Forward
            h_ff = self.ff_layers[i](h)
            h_ff = self.dropout(h_ff)

            # Normalization
            h = self.aggregation(h_ff, h)
            h = self.norm_layers[i](h.transpose(1,2)).transpose(1,2)

        return h
