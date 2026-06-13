import torch.nn as nn

from attention.multi_head_attention import MultiHeadAttention
from feed_forward_net.feed_forward import FeedForward
from normalization.layer_normalization import LayerNorm

class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.2,
        eps=1e-5
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        self.ffn = FeedForward(
            d_model,
            d_ff
        )

        self.dropout = nn.Dropout(dropout)

        self.norm1 = LayerNorm(eps)
        self.norm2 = LayerNorm(eps)
        self.norm3 = LayerNorm(eps)

    def forward(    
        self,
        x,
        encoder_output,
        src_mask,
        tgt_mask
    ):

        ############### Multi Head Attention ##################

        x = self.norm1(x)
        attn = self.self_attention(
            x,
            x,
            x,
            tgt_mask
        )
        attn = self.dropout(attn)
        x = x + attn

        ############### Cross Attention ##################

        x = self.norm2(x)
        attn = self.self_attention(
            x,
            encoder_output,
            encoder_output,
            src_mask
        )
        attn = self.dropout(attn)
        x = x + attn

        ############### Feed Forward Network ##################
        
        x = self.norm3(x)
        ffn = self.ffn(x)
        ffn = self.dropout(ffn)
        x = x + ffn

        return x

