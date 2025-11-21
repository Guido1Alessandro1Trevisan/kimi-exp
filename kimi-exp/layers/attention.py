

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

class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt((t**2).mean(dim=-1, keepdim=True) + self.eps)
        return (t * self.weight).to(dtype)


class RotaryEmbeddings(nn.Module):
    """
    Simple RoPE: produces phase angles of shape [seq_len, dim].
    Caller takes cos/sin and broadcasts.
    """

    def __init__(self, dim, base=10000.0, scale=40.0):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for rotary embeddings"
        self.dim = dim
        self.base = base
        self.scale = scale
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


class MemoryOptimizedMLA(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_size = config.d_model
        self.num_heads = config.n_heads

        self.q_a_rank = 1536
        self.kv_a_rank = 512

        self.q_nope_dim = 128
        self.q_rope_dim = 64
        self.q_head_dim = self.q_nope_dim + self.q_rope_dim
        self.v_head_dim = 128

        self.k_pe_dim = self.q_rope_dim

        self.q_a_proj = nn.Linear(self.hidden_size, self.q_a_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_a_rank)
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

        self.kv_b_proj = nn.Linear(
            self.kv_a_rank, 
            self.num_heads * (self.q_nope_dim + self.v_head_dim), 
            bias=False
        )

        self.rope = RotaryEmbeddings(self.q_rope_dim)
        self.output = nn.Linear(self.v_head_dim * self.num_heads, self.hidden_size, bias=False)

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

        rotary_emb = self.rope(seq_len)
        cos = torch.cos(rotary_emb).view(1, 1, seq_len, -1)
        sin = torch.sin(rotary_emb).view(1, 1, seq_len, -1)
        q_pe = apply_rotary(q_pe, cos, sin)
        k_pe = apply_rotary(k_pe, cos, sin)

        k = torch.cat([k_nope, k_pe], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)

        scale = 1.0 / math.sqrt(self.q_head_dim)
        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn_weights = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.output(out)


# SwiGlu
class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.ffn_dim * 2)
        self.fc2 = nn.Linear(config.ffn_dim, config.d_model)

    def forward(self, x):
        up = self.fc1(x)
        x_v, x_g = up.chunk(2, dim=-1)
        x_out = F.silu(x_v) * x_g
        return self.fc2(x_out)
    


class DeepSeekMoE(nn.Module):

    def __init__(self):
        super().__init__()
        self.shared_experts = nn.ModuleList[(Expert() for _ in range(config.n_shared))]
        self.routed_experts = nn.ModuleList[(Expert() for _ in range(config.n_experts))]
        self.gate = nn.Linear(config.d_model, config.n_experts)
        self.aux_loss = 0.0

    def forward(self, x):
        # Shared Experts process all tokens
        shared_out = sum(expert(x) for expert in self.shared_experts)

        # Device-limited routing
        routed_logits = self.gate(x)
        probs = F.softmax(routed_logits, dim=-1)
        topk_probs, topk_indices = probs.topk(routed_logits, dim=-1)

        # Expert balance loss
        expert_counts = torch.zeros(config.n_experts, device=x.device)
        expert_counts.scatter_add_(0, topk_indices.view(-1),
                                   torch.ones_like(topk_indices.view(-1), dtype=torch.float))
        self.aux_loss += expert_counts.float().var() * 0.003
        
        # Sparse Computation
        routed_out = torch.zeros_like(x)
        for k in range(config.top_k):
            expert_mask = topk_indices[..., k]
            expert_contrib = torch.zeros_like(x)

            for expert_idx in range(config.n_experts):
                mask = (expert_mask == expert_idx)
                if mask.any():
                    expert_out = self.routed_experts[expert_idx](x[mask])
                    expert_contrib[mask] = expert_out * topk_probs[..., k][mask].unsqueeze(-1)

            routed_out += expert_contrib

        return shared_out + routed_out


class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = MemoryOptimizedMLA()
        self.norm2 = nn.LayerNorm(config.d_model)
        self.moe = DeepSeekMoE()

    def forward(self, x, past_kv=None):

        attn_out, new_kv = checkpoint(self.attn, self.norm1(x), past_kv)
        x = x + attn_out

        moe_out = checkpoint(self.moe, self.norm2(x))
        x = x + moe_out

        return x, new_kv
    

class DeepSeekV2(nn.Module):

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)


    def forward(self, input_ids):
        x = self.embed(input_ids)
        total_aux_loss = 0.0

        for block in self.blocks:
            x, _ = block(x)
            total_aux_loss += block.moe.aux_loss

        return self.lm_head(self.norm(x)), total_aux_loss