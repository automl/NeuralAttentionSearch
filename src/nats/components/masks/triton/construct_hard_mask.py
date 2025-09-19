from typing import Any

import torch

import triton
import triton.language as tl
from torch import nn



@triton.jit
def _costruct_hard_mask(EndSeqsIDX,
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

    fill_indices = tl.load(EndSeqsIDX + zh_offset + offs_n, mask=n_mask, other=0)

    # if offs_n is greater than the offs_m, we fill in the corresponding end_seqs_soft values,
    # otherwise, the entire mask values
    mask = (fill_indices[None, :] > offs_m[:, None] + start_idx) & (fill_indices != offs_n)[None, :]

    filled_value = tl.where(mask,
                            float('-inf'),
                            0)

    # if this is the last element in the mask, we use 1 - soft_values[None, :] instead,
    # TODO: since we compute the gradient values only in the backward pass, We might just ignore the checks for the
    #  following two lines since they are the same under forward pass if we endcode the hard attn values

    filled_value = tl.where(offs_n[None, :] <= offs_m[:, None] + start_idx, filled_value, float('-inf'))
    tl.store(MSK + msk_offset + offs_m[:, None] * stride_mskm + offs_n[None, :],
             filled_value, mask=(offs_m[:, None] < seqlen_m) & n_mask[None, :]
             )


def construct_hard_mask(
    end_seqs_info: torch.Tensor,
    valid_size: torch.Tensor,
    start_idx: int = 0,
):
    bsz, nheads, seqlen = end_seqs_info.shape
    msk = torch.empty([bsz, nheads, seqlen - start_idx, seqlen], device=end_seqs_info.device,
                      dtype=end_seqs_info.dtype)


    grid = lambda args: (
        triton.cdiv((end_seqs_info.shape[2] - start_idx), args['BLOCK_M']) * triton.cdiv(end_seqs_info.shape[2],
                                                                                         args['BLOCK_N']),
        end_seqs_info.shape[0] * end_seqs_info.shape[1],
        1
    )
    # TODO check if this is really necessary since we always gather 1 values to the end of end_seqs_info
    end_seqs_idx = seqlen - 1 - torch.flip(torch.cummax(torch.flip(end_seqs_info, (-1,)), -1)[1], (-1,))
    end_seqs_idx += (seqlen - valid_size.to(end_seqs_info.device))

    _costruct_hard_mask[grid](end_seqs_idx, msk, start_idx,
                              end_seqs_idx.stride(0), end_seqs_idx.stride(1), end_seqs_idx.stride(2),
                              msk.stride(0), msk.stride(1), msk.stride(2), msk.stride(3),
                              bsz, nheads, seqlen - start_idx, seqlen,
                              BLOCK_M=16, BLOCK_N=128,
                              )
    return msk
