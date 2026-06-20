import torch
import torch.nn as nn

from embedding.embedding import Embedding
from encoder.encoder import Encoder
from decoder.decoder import Decoder
from feed_forward_net.projection_layer import ProjectionLayer
from positional_encoding.sinosuidal import SinusoidalPositionalEncoding

class Transformer(nn.Module): 
    def __init__(self, d_model, heads_num, dropout, vocab_size, num_layers, src_seq_len, tgt_seq_len):
        super().__init__()
        self.d_model = d_model
        self.heads_num = heads_num
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.d_ff = d_model * 4

        self.embedding = Embedding(d_model, vocab_size)
        # self.tgt_embedding = Embedding(d_model, vocab_size)

        self.src_positional_encoding = SinusoidalPositionalEncoding(d_model, seq_len=src_seq_len, dropout=dropout)
        self.tgt_positional_encoding = SinusoidalPositionalEncoding(d_model, seq_len=tgt_seq_len, dropout=dropout)
        
        self.encoder = Encoder(d_model, num_heads=heads_num, d_ff=self.d_ff, num_layers=num_layers, dropout=dropout)
        self.decoder = Decoder(d_model, heads_num=heads_num, d_ff=self.d_ff, num_layers=num_layers, dropout=dropout)

        self.projection_layer = ProjectionLayer(d_model, vocab_size)
        
    def encode(self, x, src_mask):
        # shape x -> [batch_size, seq_len] 
        x = self.embedding(x)
        # shape x -> [batch_size, seq_len, d_model] 
        x = self.src_positional_encoding(x)
        # shape x -> [batch_size, seq_len, d_model] 
        x = self.encoder(x, src_mask)
        return x

    def decode(self, x, encoder_output, src_mask, tgt_mask):
        x = self.embedding(x)
        x = self.tgt_positional_encoding(x)
        x = self.decoder(x, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return x

    def project(self, x):
        return self.projection_layer(x)
    

    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        encoder_output = self.encode(encoder_input, encoder_mask)
        decoder_output = self.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
        return self.project(decoder_output)

def build_transformer(d_model, heads_num, dropout, vocab_size, num_layers, src_seq_len, tgt_seq_len):
    model = Transformer(d_model, heads_num, dropout, vocab_size, num_layers, src_seq_len, tgt_seq_len)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
