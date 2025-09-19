import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch.nn import functional as F


def get_pad_size(x_len: int, chunk_size: int):
    return math.ceil(x_len / chunk_size) * chunk_size - x_len


@torch.compile
def average_pooling_seqlen(x: torch.Tensor, stride: int, pad_size: int = 0, dim=-2):
    """
    This function acts as an average pooling that reduce the size of x along the N_CTX dimension by computing mean value
    of every pooling_size steps.
    Args:
        x (torch.Tensor): a tensor of shape [B, H, N_CTX, HEAD_DIM]
        stride (int): stride value for N_CTXg
        pad_size (int): how many values should we pad to the N_CTX dim
        dim (int): the dimension that we want to do the padding. We note that this value should always be negative

    Returns:
        x_pool: torch.Tensor, a tensor of shape [B, H, N_CTX // stride, HEAD_DIM]

    """
    assert dim < 0
    if pad_size != 0:
        # pad_pos = [0] * (-dim*2)
        # pad_pos[-1] = pad_size
        if dim == -1:
            pad_pos = [0, pad_size]
        else:
            pad_pos = [0, 0, 0, pad_size]
        x = F.pad(x, pad_pos, )
    return x.unflatten(dim, (-1, stride)).mean(dim)


@torch.compile
def max_pooling_seqlen(x: torch.Tensor, stride: int, pad_size: int = 0, dim=-2):
    """
    This function acts as an max pooling that reduce the size of x along the N_CTX dimension by computing max value
    of every pooling_size steps.
    Args:
        x, torch.Tensor: a tensor of shape [B, H, N_CTX, HEAD_DIM]
        stride (int): stride value for N_CTXg
        pad_size (int): how many values should we pad to the N_CTX dim
        dim (int): the dimension that we want to do the padding. We note that this value should always be negative

    Returns:
        x_pool: torch.Tensor, a tensor of shape [B, H, N_CTX // stride, HEAD_DIM]

    """
    assert dim < 0
    if pad_size != 0:
        # pad_pos = [0] * (-dim*2)
        # pad_pos[-1] = pad_size
        if dim == -1:
            pad_pos = [0, pad_size]
        else:
            pad_pos = [0, 0, 0, pad_size]
        x = F.pad(x, pad_pos, )
    return x.unflatten(dim, (-1, stride)).max(dim)[0]


class ChunkMergeType(Enum):
    MEAN = 0
    MAX = 1


ChunkMergeFuncs = {
    ChunkMergeType.MEAN: average_pooling_seqlen,
    ChunkMergeType.MAX: max_pooling_seqlen,
}

