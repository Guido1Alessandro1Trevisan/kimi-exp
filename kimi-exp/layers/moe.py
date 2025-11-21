

import torch.nn.functional as F
from torch import nn
import torch
from .model import Config
import torch.distributed as dist

# SwiGlu
class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size * 2)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        up = self.fc1(x)
        x_v, x_g = up.chunk(2, dim=-1)
        x_out = F.silu(x_v) * x_g
        return self.fc2(x_out)
    

class Gate(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor

        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty(self.n_routed_experts, self.gating_dim)
        )

        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.n_routed_experts)
        )

    def forward(self, x: torch.Tensor):
        _, _, hidden_size = x.shape

        # Device-limited routing
        x_flat = x.view(-1, hidden_size)
        logits = self.gate(x_flat)
        scores = torch.sigmoid(logits)
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        topk_weight, topk_idx = scores_for_choice.topk(self.num_experts_per_tok, dim=-1)

        # Normalize and apply router scaling factor
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight

# Deep seek Layer with context parallelism
class KimiMoE(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.world_size = dist.get_world_size()
        self.ep_rank = dist.get_rank()
        assert config.n_routed_experts % self.world_size == 0, f"Number of experts must be divisible by world size"
        self.n_local_experts = config.n_routed_experts // self.world_size
        self.num_experts_per_tok = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor

        self.expert_start_idx = self.ep_rank * self.n_local_experts
        self.experts_end_idx = (self.ep_rank + 1) * self.n_local_experts

        self.gate = Gate(config)
        self.experts = nn.ModuleList(
            [
                MLP(self.hidden_size, config.moe_intermediate_size)
                if self.expert_start_idx <= i <= self.experts_end_idx
                else None
                for i in range(config.n_routed_experts)
            ]
        )
        self.aux_loss = 0.0

        # Shared experts is just a dense network
        if config.n_shared_experts is not None:
            intermediate_size = config.n_shared_experts * config.moe_intermediate_size
            self.shared_experts = MLP(
                config=config, intermediate_size=intermediate_size
            )

    def forward(self, x: torch.Tensor):
        # Shared Experts process all tokens
        batch_size, seq_len, hidden_size = x.shape

        shared_out = self.shared_experts(x)
        topk_idx, topk_weight = self.compute_gate(x)
        x_flat = x.reshape(batch_size * seq_len, -1)


        routed_flat = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            # Not in this rank
            if expert is None:
                continue
            
            mask = (topk_idx == expert_id)
            if not mask.any():
                continue

            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            expert_in = x_flat[token_idx]
            expert_out = expert(expert_in)
            gates = topk_weight[token_idx, slot_idx]

            routed_flat.index_add_(
                0,
                token_idx,
                expert_out * gates.unsqueeze(-1),
            )

        routed = routed_flat.view(batch_size, seq_len, hidden_size)

        return shared_out + routed

