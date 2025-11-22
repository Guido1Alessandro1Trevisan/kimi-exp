

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embedding: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0
        self.tp_size = dist.get_world_size()  if dist.is_initialized() else 1
        assert num_embedding % self.tp_size == 0, "Embeddings must be divisible by the tp size"
        self.num_embeddings = num_embedding
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_idx = (self.tp_rank + 1) * self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.zeros(self.num_embeddings_per_partition, embedding_dim), bias=False)
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, params: nn.Parameter, loaded_weight: torch.Tensor):
        params_data = params.data
        shard_size = params_data.size(0)
        shard_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, shard_idx, shard_size)
        params_data.copy_(loaded_weight)

    def forward(self, x):

        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__(num_embeddings, embedding_dim)

    # Need to add the prefill
    def forward(self, x: torch.Tensor):
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, dim=-1) if self.tp_rank == 0 else None
        return logits


