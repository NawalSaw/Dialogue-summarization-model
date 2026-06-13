import torch
import torch.nn as nn

from embedding.embedding import Embedding
from encoder.encoder import Encoder
from decoder.decoder import Decoder
from feed_forward_net.projection_layer import ProjectionLayer
from positional_encoding.sinosuidal import SinusoidalPositionalEncoding

class Transformer(nn.Module): 
    def __init__(self, d_model, heads_num, dropout, vocab_size, num_layers, seq_len):
        super().__init__()
        self.d_model = d_model
        self.heads_num = heads_num
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.d_ff = d_model * 4

        self.src_embedding = Embedding(d_model, vocab_size)
        self.tgt_embedding = Embedding(d_model, vocab_size)

        self.src_positional_encoding = SinusoidalPositionalEncoding(d_model, seq_len=seq_len, dropout=dropout)
        self.tgt_positional_encoding = SinusoidalPositionalEncoding(d_model, seq_len=seq_len, dropout=dropout)
        
        self.encoder = Encoder(d_model, num_heads=heads_num, d_ff=self.d_ff, num_layers=num_layers, dropout=dropout)
        self.decoder = Decoder(d_model, heads_num=heads_num, d_ff=self.d_ff, num_layers=num_layers, dropout=dropout)

        self.projection_layer = ProjectionLayer(d_model, vocab_size)
        
    def encode(self, x, src_mask):
        x = self.src_embedding(x)
        x = self.src_positional_encoding(x)
        x = self.encoder(x, src_mask)
        return x

    def decode(self, x, encoder_output, src_mask, tgt_mask):
        x = self.tgt_embedding(x)
        x = self.tgt_positional_encoding(x)
        x = self.decoder(x, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return x

    def project(self, x):
        return self.projection_layer(x)
    

    def forward(self, encoder_input, decoder_input):

        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output, encoder_input["attention_mask"])

        output = self.projection_layer(decoder_output)

        return output

def build_transformer(d_model, heads_num, dropout, vocab_size, num_layers, seq_len):
    model = Transformer(d_model, heads_num, dropout, vocab_size, num_layers, seq_len)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
