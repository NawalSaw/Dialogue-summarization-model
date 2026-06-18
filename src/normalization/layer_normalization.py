from torch import nn
import torch 

class LayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(256))
        self.beta = nn.Parameter(torch.zeros(256))
        self.eps = eps
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta 

