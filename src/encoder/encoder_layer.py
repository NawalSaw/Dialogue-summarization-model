import torch.nn as nn

from attention.multi_head_attention import MultiHeadAttention
from normalization.layer_normalization import LayerNorm
from feed_forward_net.feed_forward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, eps=1e-5):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model,
            num_heads,
            dropout
        )

        self.ffn = FeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.norm1 = LayerNorm(eps)
        self.norm2 = LayerNorm(eps)

    def forward(self, x, mask):
        residual = x
        x = self.norm1(x)
        attn = self.dropout(self.attention(x, x, x, mask))

        x = residual + attn

        # Feed Forward
        residual = x
        x = self.norm2(x)
        ff = self.dropout(self.ffn(x))

        x = residual + ff

        return x