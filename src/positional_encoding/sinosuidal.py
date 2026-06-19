import torch
import math

class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = torch.nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)

        # shape div_term -> (d_model/2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term) # even indices
        pe[:, 1::2] = torch.cos(position * div_term) # odd indices

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape:
        # (batch_size, seq_len, d_model)

        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :].requires_grad_(False)
        return self.dropout(x)