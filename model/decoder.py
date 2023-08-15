import torch.nn as nn


from model.layers import ScaledDotProductAttention

class MHADecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):
        super().__init__()
    
        self.linear_q = nn.Linear(3 * embedding_dim, embedding_dim)  # Query (Context embedding)
        self.linear_k = nn.Linear(embedding_dim, embedding_dim)      # Key (Nodes embedding)
        self.linear_v = nn.Linear(embedding_dim, embedding_dim)      # Value (Nodes embedding)
        self.linear_o = nn.Linear(embedding_dim, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.num_heads = num_heads

        # Weight Initalization
        nn.init.xavier_uniform_(self.linear_q1.weight)
        nn.init.xavier_uniform_(self.linear_k1.weight)
        nn.init.xavier_uniform_(self.linear_v1.weight)
        nn.init.xavier_uniform_(self.linear_q2.weight)
        nn.init.xavier_uniform_(self.linear_k2.weight)

    def forward(self, context_emb, nodes_emb, mask=None):
        batch_size = nodes_emb.shape[0]
        num_nodes = nodes_emb.shape[1]
        node_emb_dim = nodes_emb.shape[2]

        # First MHA
        q, k, v = self.linear_q1(context_emb), self.linear_k1(nodes_emb), self.linear_v1(nodes_emb)

        q = q.unsqueeze(1)\
            .repeat(1, num_nodes, 1)
        q = q.reshape(batch_size, num_nodes, self.num_heads, node_emb_dim // self.num_heads) # (batch_size, graph_size, num_heads, emb_per_heads)
        k = k.reshape(batch_size, num_nodes, self.num_heads, node_emb_dim // self.num_heads) # (batch_size, graph_size, num_heads, emb_per_heads)
        v = v.reshape(batch_size, num_nodes, self.num_heads, node_emb_dim // self.num_heads) # (batch_size, num_heads, graph_size, emb_per_heads)

        y = ScaledDotProductAttention()(q, k, v, mask)
        y = y.reshape(batch_size, num_nodes, node_emb_dim) # Concatenate value from each head
        
        y = self.linear_o(y).squeeze()
        # Clipping within [-10, 10]
        y = 10 * self.tanh(y)

        # Masking 
        y = y.masked_fill(mask == 1, float('-inf'))

        # Softmax 
        y = self.softmax(y)      
        return y
    
class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):
        super().__init__()
    
        self.linear_q = nn.Linear(3 * embedding_dim, embedding_dim)  # Query (Context embedding)
        self.linear_k = nn.Linear(embedding_dim, embedding_dim)      # Key (Nodes embedding)
        self.linear_v = nn.Linear(embedding_dim, embedding_dim)      # Value (Nodes embedding)
        self.linear_o = nn.Linear(embedding_dim, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.num_heads = num_heads

        # Weight Initalization
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.linear_o.weight)

    def forward(self, nodes_embs):
        raise NotImplemented

