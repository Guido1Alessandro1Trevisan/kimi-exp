

import torch
import torch.nn as nn
from layers.norm import RMSNorm
import math
from .model import Config


class RotaryEmbeddings(nn.Module):
    """
    Simple RoPE: produces phase angles of shape [seq_len, dim].
    Caller takes cos/sin and broadcasts.
    """

    def __init__(self, dim, config: Config):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for rotary embeddings"
        self.dim = dim
        self.base = config.rope_theta
        self.scale = config.rope_scaling_factor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int):
        device = self.inv_freq.device
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype) / self.scale
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [T, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)           # [T, dim]
        return emb


def rotate_half(x: torch.Tensor):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class MLA(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.q_a_rank = 1536
        self.kv_a_rank = 512

        self.q_nope_dim = 128
        self.q_rope_dim = 64
        self.q_head_dim = self.q_nope_dim + self.q_rope_dim
        self.v_head_dim = 128

        self.k_pe_dim = self.q_rope_dim

        self.q_a_proj = nn.Linear(self.hidden_size, self.q_a_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_a_rank)
        # Down Projection LORA
        self.q_b_proj = nn.Linear(
            self.q_a_rank, 
            self.num_heads * self.q_head_dim, 
            bias=False
        )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_a_rank + self.k_pe_dim,
            bias=False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_a_rank)
        # Down Projection Lora
        self.kv_b_proj = nn.Linear(
            self.kv_a_rank, 
            self.num_heads * (self.q_nope_dim + self.v_head_dim), 
            bias=False
        )

        self.rotary_emb = RotaryEmbeddings(self.q_rope_dim, config)
        self.o_proj = nn.Linear(self.v_head_dim * self.num_heads, self.hidden_size, bias=False)

    def forward(self, h):
        
        batch_size, seq_len, _ = h.shape

        q_a = self.q_a_proj(h)
        q_a = self.q_a_layernorm(q_a)
        q = self.q_b_proj(q_a)
        q: torch.Tensor = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = q.split([self.q_nope_dim, self.q_rope_dim], dim=-1)

        kv_a: torch.Tensor = self.kv_a_proj_with_mqa(h)
        kv_nope, k_pe = kv_a.split([self.kv_a_rank, self.k_pe_dim], dim=-1)
        kv_nope = self.kv_a_layernorm(kv_nope)
        kv = self.kv_b_proj(kv_nope)
        kv: torch.Tensor = kv.view(batch_size, seq_len, self.num_heads, self.q_nope_dim + self.v_head_dim).transpose(1, 2) 
        k_nope, v = kv.split([self.q_nope_dim, self.v_head_dim], dim=-1)
        k_pe = k_pe.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, self.k_pe_dim)

        rotary_emb = self.rotary_emb(seq_len)
        cos = torch.cos(rotary_emb).view(1, 1, seq_len, -1)
        sin = torch.sin(rotary_emb).view(1, 1, seq_len, -1)
        q_pe = apply_rotary(q_pe, cos, sin)
        k_pe = apply_rotary(k_pe, cos, sin)

        k = torch.cat([k_nope, k_pe], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)

        scale = 1.0 / math.sqrt(self.q_head_dim)
        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
        mask = torch.triu(attn.new_full((seq_len, seq_len), -float("inf")), diagonal=1)
        attn = attn + mask
        attn_weights = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)

