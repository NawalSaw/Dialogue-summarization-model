from torch import nn

from decoder.decoder_layer import DecoderLayer
from normalization.layer_normalization import LayerNorm

class Decoder(nn.Module):
    def __init__(self, d_model, heads_num, d_ff, num_layers, dropout=0.1, eps=1e-5):
        super().__init__()

        self.norm = LayerNorm(eps)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    heads_num,
                    d_ff,
                    dropout,
                    eps
                )
                for _ in range(num_layers)
            ]
        )
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        
        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                src_mask,
                tgt_mask
            )
        
        return self.norm(x)

