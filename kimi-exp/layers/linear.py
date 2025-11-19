

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F 

def divide(numerator, denomenator):
    assert numerator % denomenator == 0, "Numerator must be divisible by denomenator"
    return denomenator // numerator


class LinearBase(nn.Module):

    def __init__(self, input_size, output_size, bias = False, tp_dim = None):
        super().__init__(self)

        self.tp_dim = tp_dim
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        self.weights = nn.Parameter(torch.zeros(output_size, input_size))
        self.weights.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.bias.weight_lodaer = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        raise NotImplemented


class ReplicateLinear(LinearBase):

    def __init__(self, input_size, output_size, bias):
        super().__init__(self, input_size, output_size, bias, None)

    def weight_loader(self, params: nn.Parameter, loaded_weights: torch.Tensor):
        params.data.copy_(loaded_weights)

    def forward(self, x):
        return F.linear(x, self.weights, self.bias)


class ColumnLinearParallel(LinearBase):

    def __init__(self, input_size, output_size, bias):
        tp_size = dist.get_world_size()
        super().__init__(self, input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, params: nn.Parameter, loaded_weights: torch.Tensor):
        params_data = params.data
        shard_size = params_data.size(self.tp_dim)
        shard_offset = shard_size * self.tp_rank
        loaded_weights = loaded_weights.narrow(self.tp_dim, shard_offset, shard_size)
        params_data.copy_(loaded_weights)

    def forward(self, x):
        return F.linear(x, self.weights, self.bias)
        

        


        