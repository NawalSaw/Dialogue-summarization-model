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
        dropout=0.1,
        eps=1e-5
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        self.ffn = FeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.norm1 = LayerNorm(eps)
        self.norm2 = LayerNorm(eps)
        self.norm3 = LayerNorm(eps)

    def forward(    
        self,
        x, # shape: (batch_size, seq_len, d_model)
        encoder_output, # shape: (batch_size, seq_len, d_model)
        src_mask, # shape: (batch_size, 1, seq_len, seq_len)
        tgt_mask # shape: (batch_size, 1, seq_len, seq_len)
    ):

        ############### Self Attention ##################

        residual_1 = x
        x = self.norm1(x)
        attn = self.self_attention(
            x,
            x,
            x,
            tgt_mask
        )
        attn = self.dropout(attn)
        x = residual_1 + attn
        
        ############### Cross Attention ##################

        residual_2 = x
        x = self.norm2(x)
        attn = self.cross_attention(
            x,
            encoder_output,
            encoder_output,
            src_mask
        )
        attn = self.dropout(attn)
        x = residual_2 + attn
        # print("Cross attention output:", x.abs().mean().item())
        ############### Feed Forward Network ##################
        # cross_attn_output_std = x.std()
        # print("Cross attention output std:", cross_attn_output_std)
        
        residual_3 = x
        x = self.norm3(x)
        ffn = self.ffn(x)
        ffn = self.dropout(ffn)
        x = residual_3 + ffn
        # print("Feed forward output:", x.abs().mean().item())

        return x

