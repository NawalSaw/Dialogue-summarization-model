import torch
from torch import nn
import math

class Embedding(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.embed_dim) # shape (batch_size, seq_len, embed_dim)
