from torch import nn
from encoder.encoder_layer import EncoderLayer
from normalization.layer_normalization import LayerNorm

class Encoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        dropout=0.2,
        eps=1e-5
    ):
        super().__init__()

        self.norm = LayerNorm(eps)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout,
                    eps
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, src_mask):

        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)
