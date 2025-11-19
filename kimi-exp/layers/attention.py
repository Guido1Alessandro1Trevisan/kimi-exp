

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
import math

class Config:

    def __init__(self):
        self.vocab_size = 32000
        self.d_model = 5120
        self.n_layers = 2
        self.n_heads = 8
        self.d_kv_comp = 128
        self.d_rope = 16
        self.n_experts = 32
        self.n_shared = 2
        self.top_k = 2
        self.seq_len = 256
        self.batch_size = 1
        self.ffn_dim = 384
        self.device_groups = 4


config = Config()

class RotaryEmbeddings(nn.Module):

    def __init__(self, dim, scale=40):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for rotary embeddings"
        self.dim = dim
        self.inv_freq = 1.0 / (1000 ** (torch.arange(0, dim // 2, 2).float() / (dim / 2)))
        self.register_buffer("inv_freq", self.inv_freq)
        self.scale = 40

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq).type_as(self.inv_freq) / self.scale
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)
    
def rotate_half(x: torch.Tensor):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x, cos, sin):

    x_rot = (x_rot * cos) + (rotate_half(x) * sin)

    return x_rot


class MemoryOptimizedMLA(nn.Module):

    def __init__(self):
        super().__init__()
        self.d_head = config.d_model // config.n_heads
        self.split_dim = self.d_head - config.d_rope

        self.W_dkv = nn.Linear(config.d_model, config.d_kv_comp)
        self.W_dq = nn.Linear(config.d_model, config.d_kv_comp)

        self.W_uk = nn.Linear(config.d_kv_comp, config.n_heads * self.split_dim)
        self.W_uv = nn.Linear(config.d_kv_comp, config.n_heads * self.d_head)
        self.W_uq = nn.Linear(config.d_kv_comp, config.n_heads * self.split_dim)

        self.W_qr = nn.Linear(config.d_kv_comp, config.n_heads * config.d_rope)
        self.W_kr = nn.Linear(config.d_kv_comp, config.n_heads * config.d_rope)

        self.rotary = RotaryEmbeddings(config.n_heads * self.d_head, config.d_model)
        self.output = nn.Linear(config.n_heads * self.d_head, config.d_model)

    def forward(self, h, past_kv=None):
        batch_size, seq_len, _ = h.shape

        c_kv = self.W_dkv(h)
        k_base = self.W_uk(c_kv).view(batch_size, seq_len, config.n_heads, self.split_dim)
        k_rot = self.W_kr(k).view(batch_size, seq_len, config.n_heads, config.d_rope)
        v = self.W_uv(c_kv).view(batch_size, seq_len, config.n_heads, self.d_head)

        c_q = self.W_dq(h)
        q_base = self.W_uq(c_q).view(batch_size, seq_len, config.n_heads, self.split_dim)
        q_rot = self.W_qr(c_q).view(batch_size, seq_len, config.n_heads, config.d_rope)

        rotary_emb = self.rotary(seq_len)
        cos = torch.cos(rotary_emb).view(1, seq_len, 1, -1)
        sin = torch.sin(rotary_emb).view(1, seq_len, 1, -1)

        q_rot = apply_rotary(q_rot, cos, sin)
        k_rot = apply_rotary(k_rot, cos, sin)

        q = torch.cat([q_base, q_rot], dim=-1)
        k = torch.cat([k_base, k_rot], dim=-1)

        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v)

        return self.output(out.contiguous().view(batch_size, seq_len, -1)), (c_kv, k_rot)
    
    


