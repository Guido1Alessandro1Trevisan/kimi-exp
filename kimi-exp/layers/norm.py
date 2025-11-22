
import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt((t**2).mean(dim=-1, keepdim=True) + self.eps)
        return (t * self.weight).to(dtype)