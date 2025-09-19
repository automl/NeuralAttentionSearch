import copy
from typing import Optional

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from torchtune.modules.peft import AdapterModule
from torch import nn


class RMSNorm(torch.nn.Module):
    # For llama3
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False, for GPT"""

    def __init__(self, ndim, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class LLAMA3FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_hidden_dim: Optional[int] = None,
        ffn_dim_multiplier: Optional[float] = None,
        on_ddp: bool = False,
    ):
        super().__init__()
        if ffn_hidden_dim is not None:
            hidden_dim = ffn_hidden_dim
        else:
            hidden_dim = int(2 * hidden_dim / 3)
            # custom dim factor multiplier
            if ffn_dim_multiplier is not None:
                hidden_dim = int(ffn_dim_multiplier * hidden_dim)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        """
        if on_ddp:
            self.w1 = ColumnParallelLinear(
                dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
            )
            self.w2 = RowParallelLinear(
                hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
            )
            self.w3 = ColumnParallelLinear(
                dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
            )
        else:
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(dim, hidden_dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        """

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GPTFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim_multiplier: int,
        bias: bool = False,
        dropout: float = 0.0,
        on_ddp: bool = False
    ):
        super().__init__()
        self.c_fc = nn.Linear(dim, ffn_dim_multiplier * dim, bias=bias)
        self.c_proj = nn.Linear(ffn_dim_multiplier * dim, dim, bias=bias)
        """
        if on_ddp:
            self.c_fc = ColumnParallelLinear(dim, ffn_dim_multiplier * dim,
                                             bias=bias, gather_output=False, init_method=lambda x: x)
            self.c_proj = ColumnParallelLinear(ffn_dim_multiplier * dim, dim,
                                               bias=bias, gather_output=False, init_method=lambda x: x)
        else:
            self.c_fc = nn.Linear(dim, ffn_dim_multiplier * dim, bias=bias)
            self.c_proj = nn.Linear(ffn_dim_multiplier * dim, dim, bias=bias)
        """
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class NormLayerAdapter(nn.Module, AdapterModule):
    """
    implementaiton based on the ln LNTuningLayer from peft:
     https://github.com/huggingface/peft/blob/main/src/peft/tuners/ln_tuning/layer.py
    """
    def __init__(self, base_layer: nn.Module):
        super(NormLayerAdapter, self).__init__()
        self.base_layer = base_layer
        self.tuning_layer = copy.deepcopy(base_layer)
        self.disabled = False

        for n, p in self.base_layer.named_parameters():
            p.requires_grad = False

    def regenerate_tuning_layer(self):
        self.tuning_layer = copy.deepcopy(self.base_layer)
        for n, p in self.base_layer.named_parameters():
            p.requires_grad = True

    def adapter_params(self) -> list[str]:
        adapter_params = []
        for n, p in self.tuning_layer.named_parameters():
            if p.requires_grad:
                adapter_params.append(f'tuning_layer.{n}')

        return adapter_params

    def forward(self, x):
        if self.disabled:
            return self.base_layer(x)
        else:
            return self.tuning_layer(x)
