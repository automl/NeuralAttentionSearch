import torch
from typing import Any
from torch import nn

import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_M': 32, }, num_stages=4, num_warps=8),

        triton.Config({'BLOCK_M': 16, }, num_stages=4, num_warps=8),

        triton.Config({'BLOCK_M': 32, }, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 16, }, num_stages=4, num_warps=2),

        triton.Config({'BLOCK_M': 32, }, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16, }, num_stages=2, num_warps=8),

        triton.Config({'BLOCK_M': 32, }, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, }, num_stages=2, num_warps=4),

        triton.Config({'BLOCK_M': 32, }, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 16, }, num_stages=5, num_warps=2),
    ]


def get_autotune_config():
    return get_cuda_autotune_config()


@triton.jit
def cummax(v0, i0, j0, v1, i1, j1):
    gt = v0 > v1
    return tl.where(gt, v0, v1), tl.where(gt, i0, i1), tl.where(gt, j0, j1)


"""
@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['bsz', 'nheads','seqlen'],
)
"""
@triton.jit
def _reversed_cummax_forward(EndSeqsHard, EndSeqsSoft,
                             OutSoft, OutIndex,
                             stride_xz, stride_xh, stride_xn,  #
                             stride_oz, stride_oh, stride_on,  #
                             bsz, nheads, seqlen,
                             BLOCK_M: tl.constexpr,
                             BLOCK_N: tl.constexpr):
    range_n = tl.arange(0, BLOCK_N)
    range_m = tl.arange(0, BLOCK_M)
    pid = tl.program_id(axis=0) * BLOCK_M  # We use a 1D launch grid so axis is 0.

    off_z = pid // nheads
    off_h = pid % nheads
    zh_offset = off_z.to(tl.int64) * stride_xz + off_h.to(tl.int64) * stride_xh

    off_m = zh_offset + range_m[:, None] * stride_xh
    mask_n = range_n[None, :] < seqlen
    mask_m = off_m < bsz * nheads * stride_xh

    end_seqs_hard = tl.load(
        EndSeqsHard + off_m + range_n[None, :], mask=mask_m & mask_n,
        other=0
    )
    end_seqs_soft = tl.load(
        EndSeqsSoft + off_m + range_n[None, :], mask=mask_m & mask_n,
        other=0
    )
    _, o_soft, o_idx = tl.associative_scan(
        (end_seqs_hard, end_seqs_soft, tl.broadcast_to(range_n[None, :], (BLOCK_M, BLOCK_N))),
        axis=-1,
        combine_fn=cummax,
        reverse=True
    )

    off_m_store = zh_offset + range_m[:, None] * stride_oh
    mask_m_store = (off_m_store < bsz * nheads * stride_oh)

    tl.store(
        OutIndex + off_m_store + range_n[None, :], o_idx,
        mask=mask_m_store & mask_n,
    )
    tl.store(
        OutSoft + off_m_store + range_n[None, :], o_soft,
        mask=mask_m_store & mask_n,
    )

#
# class CumMaxReverse(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx: Any, end_seqs_hard: torch.Tensor, end_seqs_soft: torch.Tensor):
#         out_idx = torch.empty(*end_seqs_hard.shape, device=end_seqs_hard.device, dtype=torch.long)
#         out_soft = torch.empty_like(end_seqs_soft)
#         assert end_seqs_hard.is_cuda and out_idx.is_cuda
#
#         bsz, nheads, seqlen = end_seqs_hard.shape
#         BLOCK_N = triton.next_power_of_2(seqlen)
#         grid = lambda args: (triton.cdiv(bsz * nheads, args['BLOCK_M']),)
#
#         _reversed_cummax_forward[grid](end_seqs_hard, end_seqs_soft, out_soft, out_idx,
#                                        end_seqs_hard.stride(0), end_seqs_hard.stride(1), end_seqs_hard.stride(2),
#                                        out_soft.stride(0), out_soft.stride(1), out_soft.stride(2),
#                                        bsz, nheads, seqlen,
#                                        BLOCK_M=4,
#                                        BLOCK_N=BLOCK_N)
#         ctx.save_for_backward(out_idx)
#         return out_idx, out_soft
#
#     @staticmethod
#     def backward(ctx: Any, grad_idx, grad_soft) -> Any:
#         """
#         For backpropagation, since only out_soft is differentiable, we could ignore the parts for out_idx and only
#         focusing on out_soft, This results us a jacobian deteriend by out_idx:
#         assuming that we have a out_idx = [2,2,2, 4, 4], then its jacobian should be
#         [0, 0, 1, 0, 0,
#          0, 0, 1, 0, 0,
#          0, 0, 1, 0, 0,
#          0, 0, 0, 0, 1,
#          0, 0, 0, 0, 1],
#          assuming that do is [d0, d1, d2, d3, d4]
#          This result us a final grad as
#          [0, 0, d1+d2+d3, 0, d4+d5]
#          should also be implemented here?
#          https://github.com/pytorch/pytorch/pull/30881/files#diff-25ec2c1108ee03e2167622588ec31d167897ef1cccb12a4cfe77eb98777316daR434
#         """
#         out_idx = ctx.saved_tensors[0]
#         grad_ = torch.zeros_like(grad_soft)
#         return None, grad_.scatter_add(-1, out_idx, grad_soft)
#
#
# def cummax_reverse(end_seqs_hard: torch.Tensor, end_seqs_soft: torch.Tensor):
#     return CumMaxReverse.apply(end_seqs_hard, end_seqs_soft)


def cummax_reverse(end_seqs_hard: torch.Tensor, end_seqs_soft: torch.Tensor):
    # we don not need the backward here, everything will be implemented in construct_soft_mask!!!
    out_idx = torch.empty(*end_seqs_hard.shape, device=end_seqs_hard.device, dtype=torch.long)
    out_soft = torch.empty_like(end_seqs_soft)
    assert end_seqs_hard.is_cuda and out_idx.is_cuda

    bsz, nheads, seqlen = end_seqs_hard.shape
    BLOCK_N = triton.next_power_of_2(seqlen)
    grid = lambda args: (triton.cdiv(bsz * nheads, args['BLOCK_M']),)

    _reversed_cummax_forward[grid](end_seqs_hard, end_seqs_soft, out_soft, out_idx,
                                   end_seqs_hard.stride(0), end_seqs_hard.stride(1), end_seqs_hard.stride(2),
                                   out_soft.stride(0), out_soft.stride(1), out_soft.stride(2),
                                   bsz, nheads, seqlen,
                                   BLOCK_M=4,
                                   BLOCK_N=BLOCK_N)
    return out_idx, out_soft
