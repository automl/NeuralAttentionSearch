from typing import Any

import torch

import triton
import triton.language as tl
from torch import nn


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, }, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, }, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, }, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, }, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, }, num_stages=4, num_warps=8),

        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, }, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, }, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, }, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, }, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, }, num_stages=5, num_warps=4),

        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, }, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, }, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, }, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, }, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, }, num_stages=2, num_warps=4),

        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, }, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, }, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, }, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, }, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, }, num_stages=5, num_warps=8),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['bsz', 'nheads', 'seqlen_m'],
)
@triton.jit
def _costruct_soft_mask_forward(EndSeqsIDX, EndSeqSoft, CollectedValues,
                                MSK, start_idx: int,
                                stride_xz, stride_xh, stride_xl,  #
                                stride_mskz, stride_mskh, stride_mskm, stride_mskn,  #
                                bsz, nheads, seqlen_m, seqlen_n,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_mn = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // nheads
    off_h = off_hz % nheads

    num_pid_m = tl.cdiv(seqlen_m, BLOCK_M)

    start_m = start_mn % num_pid_m
    start_n = start_mn // num_pid_m

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    zh_offset = off_z.to(tl.int64) * stride_xz + off_h.to(tl.int64) * stride_xh
    msk_offset = off_z.to(tl.int64) * stride_mskz + off_h.to(tl.int64) * stride_mskh

    n_mask = offs_n < seqlen_n

    soft_values = tl.load(EndSeqSoft + zh_offset + offs_n, mask=n_mask, other=0)
    collected_values = tl.load(CollectedValues + zh_offset + offs_n, mask=n_mask, other=0)
    fill_indices = tl.load(EndSeqsIDX + zh_offset + offs_n, mask=n_mask, other=0)

    # if offs_n is greater than the offs_m, we fill in the corresponding end_seqs_soft values,
    # otherwise, the entire mask values
    filled_value = tl.where(fill_indices[None, :] >= offs_m[:, None] + start_idx,
                            tl.log(collected_values[None, :]),
                            tl.log(soft_values[None, :]))

    # if this is the last element in the mask, we use 1 - soft_values[None, :] instead,
    # TODO: since we compute the gradient values only in the backward pass, We might just ignore the checks for the
    #  following two lines since they are the same under forward pass if we endcode the hard attn values
    filled_value = tl.where(fill_indices == seqlen_n - 1, tl.log(1 - soft_values[None, :]), filled_value)
    filled_value = tl.where(offs_n[None, :] == offs_m[:, None] + start_idx, 0., filled_value)

    filled_value = tl.where(offs_n[None, :] <= offs_m[:, None] + start_idx, filled_value, float('-inf'))
    tl.store(MSK + msk_offset + offs_m[:, None] * stride_mskm + offs_n[None, :],
             filled_value, mask=(offs_m[:, None] < seqlen_m) & n_mask[None, :]
             )


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['bsz', 'nheads', 'seqlen_m'],
)
@triton.jit
def _costruct_soft_mask_backward(
    EndSeqsIDX, EndSeqsSoft, CollectedValues,
    GradSeqSoft, GradCollectedValues,
    GradMSK, start_idx: int,
    stride_xz, stride_xh, stride_xl,  #
    stride_mskz, stride_mskh, stride_mskm, stride_mskn,  #
    bsz, nheads, seqlen_m, seqlen_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_n = tl.program_id(0) * BLOCK_N
    off_hz = tl.program_id(1)

    off_z = off_hz // nheads
    off_h = off_hz % nheads
    zh_offset = off_z.to(tl.int64) * stride_xz + off_h.to(tl.int64) * stride_xh
    msk_offset = off_z.to(tl.int64) * stride_mskz + off_h.to(tl.int64) * stride_mskh

    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_mask = offs_n < seqlen_n

    # Since log are applied everywhere, we compute the inverse here

    soft_values = 1. / tl.load(EndSeqsSoft + zh_offset + offs_n, mask=n_mask, other=0)
    collected_values = 1. / tl.load(CollectedValues + zh_offset + offs_n, mask=n_mask, other=0)
    fill_indices = tl.load(EndSeqsIDX + zh_offset + offs_n, mask=n_mask, other=0)

    # we check the last index
    is_last_idx = (fill_indices == seqlen_n - 1)

    # TODO check necessary data type of this arguments
    grad_soft = tl.zeros([BLOCK_N], dtype=tl.float32)
    grad_collect = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Since the upper triangle is 0 tensor, we could start from the lower part of the tensor
    for start_m in range(0, tl.cdiv(seqlen_m - start_n, BLOCK_M)):
        # for start_m in range(0, tl.cdiv(seqlen_m, BLOCK_M)):
        offs_m = start_m * BLOCK_M + start_n + tl.arange(0, BLOCK_M)

        grad_msk = tl.load(GradMSK + msk_offset + offs_m[:, None] * stride_mskm + offs_n[None, :],
                           mask=(offs_m[:, None] < seqlen_m) & n_mask[None, :],
                           other=0
                           )

        # we first compute the gradients for the upper triangle. forwards are log(collected_values)
        # We store the gradients towards grad_collect, which can be further passed to the grad_soft

        grad_collect += tl.sum(
            tl.where(
                (fill_indices[None, :] >= offs_m[:, None] + start_idx) & (~is_last_idx),
                grad_msk * tl.broadcast_to(collected_values[None, :], [BLOCK_M, BLOCK_N]),
                0
            ), 0
        )

        # the next is for the last index, grad for log(1-soft)
        grad_soft += tl.sum(
            tl.where(
                (tl.broadcast_to(is_last_idx[None, :], (BLOCK_M, BLOCK_N))),
                -(grad_msk * tl.broadcast_to(soft_values[None, :], [BLOCK_M, BLOCK_N])),
                0
            ), 0
        )

        # finally, we have, grad for log(soft)
        grad_soft += tl.sum(
            tl.where(
                fill_indices[None, :] >= offs_m[:, None] + start_idx,
                0,
                grad_msk * tl.broadcast_to(soft_values[None, :], [BLOCK_M, BLOCK_N])
            ), 0
        )

    tl.store(GradSeqSoft + zh_offset + offs_n, grad_soft, mask=n_mask)
    tl.store(GradCollectedValues + zh_offset + offs_n, grad_collect, mask=n_mask)


class ConstructSoftMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                end_seqs_idx: torch.Tensor,
                end_seqs_soft: torch.Tensor,
                end_seqs_soft_collected: torch.Tensor,
                start_idx: int = 0,
                ):
        bsz, nheads, seqlen = end_seqs_idx.shape

        msk = torch.empty([bsz, nheads, seqlen - start_idx, seqlen], device=end_seqs_soft.device,
                          dtype=end_seqs_soft.dtype)

        grid = lambda args: (
            triton.cdiv((end_seqs_idx.shape[2] - start_idx), args['BLOCK_M']) * triton.cdiv(end_seqs_idx.shape[2],
                                                                                            args['BLOCK_N']),
            end_seqs_idx.shape[0] * end_seqs_idx.shape[1],
            1
        )

        _costruct_soft_mask_forward[grid](end_seqs_idx, end_seqs_soft, end_seqs_soft_collected, msk, start_idx,
                                          end_seqs_idx.stride(0), end_seqs_idx.stride(1), end_seqs_idx.stride(2),
                                          msk.stride(0), msk.stride(1), msk.stride(2), msk.stride(3),
                                          bsz, nheads, seqlen - start_idx, seqlen,
                                          # BLOCK_M=128, BLOCK_N=64,
                                          )
        ctx.save_for_backward(end_seqs_idx, end_seqs_soft, end_seqs_soft_collected)
        ctx.start_idx = start_idx
        return msk

    @staticmethod
    def backward(ctx: Any, grad_msk) -> Any:
        """
        Backward for constructing the maks. Here, we compute the  gradients to end_seqs_soft and end_seqs_soft_collected
        individually. Where the end_seqs_soft is only responsible for the lower part of the attention maps (Those are
        not filled by grad_soft_collected )
        """
        end_seqs_idx, end_seqs_soft, end_seqs_soft_collected = ctx.saved_tensors
        start_idx = ctx.start_idx

        grad_soft = torch.zeros_like(end_seqs_soft)
        grad_collected = torch.zeros_like(end_seqs_soft_collected)

        grid = lambda args: (
            triton.cdiv(end_seqs_idx.shape[2], args['BLOCK_N']),
            end_seqs_idx.shape[0] * end_seqs_idx.shape[1],
            1
        )

        _costruct_soft_mask_backward[grid](
            end_seqs_idx, end_seqs_soft, end_seqs_soft_collected,
            grad_soft, grad_collected,
            grad_msk, start_idx,
            end_seqs_idx.stride(0), end_seqs_idx.stride(1), end_seqs_idx.stride(2),
            grad_msk.stride(0), grad_msk.stride(1), grad_msk.stride(2), grad_msk.stride(3),
            grad_msk.shape[0], grad_msk.shape[1], grad_msk.shape[2], grad_msk.shape[3],
            # BLOCK_M=256, BLOCK_N=64,
        )
        # add the gradients of collected values to soft
        grad_soft = grad_soft.scatter_add(-1, end_seqs_idx, grad_collected)
        return None, grad_soft, None, None


def construct_soft_mask(end_seqs_idx: torch.Tensor,
                        end_seqs_soft: torch.Tensor,
                        end_seqs_soft_collected: torch.Tensor,
                        start_idx: int = 0, ):
    return ConstructSoftMask.apply(end_seqs_idx, end_seqs_soft, end_seqs_soft_collected, start_idx)
