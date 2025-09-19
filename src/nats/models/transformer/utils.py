
import torch
import math
from enum import Enum


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
            .expand(bs, n_kv_heads, n_rep, slen, head_dim)
            .reshape(bs, n_kv_heads * n_rep,slen, head_dim)
    )


def repeat_masks(mask: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This function is similar to repeat_kv, however, we will expand the second dimension
    """
    bs, n_kv_heads, slen1, slen2 = mask.shape
    if n_rep == 1 or n_kv_heads == 1:
        return mask
    return (
        mask[:, :, None, :, :]
            .expand(bs, n_kv_heads, n_rep, slen1, slen2)
            .reshape(bs, n_kv_heads * n_rep, slen1, slen2)
    )


def repeat_end_seqs_values(end_seqs: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return end_seqs

    bs, n_kv_heads, slen = end_seqs.shape
    return (
        end_seqs[:, :, None, :, :]
            .expand(bs, n_kv_heads, n_rep, slen)
            .reshape(bs, n_kv_heads * n_rep, slen)
    )

