import torch.nn as nn


from model.layers import ScaledDotProductAttention

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