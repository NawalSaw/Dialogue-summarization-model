import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads

        self.head_dim = d_model // num_heads
        self.attn_dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask=None, attn_dropout=None):

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = attn_dropout(attention_weights)
        return torch.matmul(attention_weights, value)

    def forward(self, query, key, value, mask=None):
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        # We do this because we want to compute the attention for each head separately and allow each head to look for its own sequence pattern in given dimensions

        q = q.reshape(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        attention = MultiHeadAttention.attention(q, k, v, mask, self.attn_dropout)
        
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, d_model)
        attention = attention.transpose(1, 2).contiguous()

        attention = attention.view(
            attention.size(0),
            attention.size(1),
            self.d_model
        )
                
        return self.W_o(attention)

        