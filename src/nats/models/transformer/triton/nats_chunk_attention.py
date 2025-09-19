import copy
from typing import Any
import math

import torch
from torch.nn import functional as F

import triton
import triton.language as tl
from einops import reduce, repeat, rearrange

from nats.utils import check_fp16_dtype
import math

fp_16_type = check_fp16_dtype()

fp16_dtype = tl.float16 if fp_16_type == 'float16' else tl.bfloat16


@torch.compile
def average_pooling_seqlen(x: torch.Tensor, stride: int, pad_size: int = 0, dim=-2):
    """
    This function acts as an average pooling that reduce the size of x along the N_CTX dimension by computing mean value
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
    return x.unflatten(dim, (-1, stride)).mean(dim)


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, n_valid_blocks, q,  #
                    K_block_ptr, V_block_ptr, Seqinfo_block_ptr, SeqsIdx_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    CHUNK_NAtS: tl.constexpr,
                    LOCAL_SEQ_MAX_LEN: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr,
                    N_NAtS_BLOCK_PER_N: tl.constexpr,
                    fp8_v: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # Current, we only consider the cases where CHUNK_NAtS >= BLOCK_N
    # This ensures that the block within each kernel

    # loop over k, v and update accumulator

    start_mM = start_m * BLOCK_M
    end_n_offset = tl.arange(0, N_NAtS_BLOCK_PER_N) * tl.minimum(CHUNK_NAtS, BLOCK_N)

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        advance_nats = start_n // CHUNK_NAtS

        fill_indices_ptr_ = tl.advance(SeqsIdx_block_ptr, (advance_nats,))
        sliding_window_tokens_ptr_ = tl.advance(Seqinfo_block_ptr, (advance_nats,))

        offs_n_ = start_n + offs_n

        fill_indices = tl.load(fill_indices_ptr_, boundary_check=(0,), padding_option='zero')
        # fill_indices_scaled is the starting of the block. However, we want to preserve until the last element of
        # of the block, which is CHUNK_NAtS

        is_sw_token = tl.load(sliding_window_tokens_ptr_, boundary_check=(0,), padding_option='zero').to(tl.int1)
        is_global_token = fill_indices == ((start_n // CHUNK_NAtS) + tl.arange(0, N_NAtS_BLOCK_PER_N))
        # the following codes check if we need to compute this block without checking the masks first, need to check
        # which approach is faster...

        fill_indices_scaled = fill_indices * CHUNK_NAtS + CHUNK_NAtS
        has_valid_indices = fill_indices_scaled > start_mM

        # the last element in the sw token must be larger than the corresponidng start_mN
        sw_token_end_idx = start_n + LOCAL_SEQ_MAX_LEN + end_n_offset + CHUNK_NAtS - 1
        has_valid_sw = sw_token_end_idx >= start_mM

        do_sw_compute = tl.sum((is_sw_token & has_valid_sw).to(tl.int32))
        #do_local_compute = tl.sum(((1 - is_sw_token.to(tl.int32)) & has_valid_indices).to(tl.int32))
        do_local_compute = tl.sum(((~is_sw_token) & has_valid_indices).to(tl.int32))
        # for debugging
        do_compute = (tl.sum(is_global_token.to(tl.int32)) + do_sw_compute + do_local_compute).to(tl.int1)

        if do_compute:
            msk_local = tl.broadcast_to(fill_indices_scaled[None, :], (BLOCK_M, N_NAtS_BLOCK_PER_N)) > offs_m[:, None]
            msk_sw = offs_m[:, None] <= sw_token_end_idx[None, :]
            msk_casual = (offs_m[:, None] >= offs_n_[None, :])
            if N_NAtS_BLOCK_PER_N == 1 or CHUNK_NAtS == 1:
                msk_locals = tl.where(is_sw_token, msk_sw, msk_local)
                msk = tl.where(is_global_token, msk_casual, msk_casual & msk_locals)
            else:
                #
                # msk_local = tl.reshape(msk_local, [BLOCK_M, 1, N_NAtS_BLOCK_PER_N])
                # msk_sw = tl.reshape(msk_sw, [BLOCK_M, CHUNK_NAtS, N_NAtS_BLOCK_PER_N])
                # msk_casual = tl.reshape(msk_casual, [BLOCK_M, CHUNK_NAtS, N_NAtS_BLOCK_PER_N])
                msk_local = tl.reshape(msk_local, [BLOCK_M, N_NAtS_BLOCK_PER_N, 1])
                msk_sw = tl.reshape(msk_sw, [BLOCK_M, N_NAtS_BLOCK_PER_N, 1])
                msk_casual = tl.reshape(msk_casual, [BLOCK_M, N_NAtS_BLOCK_PER_N, CHUNK_NAtS])

                msk_locals = tl.where(is_sw_token[None, :, None], msk_sw, msk_local)
                msk = tl.where(is_global_token[None, :, None], msk_casual, msk_casual & msk_locals)
                msk = tl.reshape(msk, [BLOCK_M, BLOCK_N])

            n_blocks_rows = tl.sum(msk.to(tl.int32), -1)

            n_valid_blocks += n_blocks_rows

            # -- compute qk ----
            k = tl.load(K_block_ptr, boundary_check=(1,), padding_option='zero')
            qk = tl.dot(q, k)
            qk = qk * qk_scale + tl.where(msk, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]

            p = tl.math.exp2(qk)

            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            # -- update output accumulator --
            acc = acc * alpha[:, None]
            # update acc
            v = tl.load(V_block_ptr, boundary_check=(0,), padding_option='zero')
            if fp8_v:
                p = p.to(tl.float8e5)
            else:
                p = p.to(fp16_dtype)
            acc = tl.dot(p, v, acc)
            # update m_i and l_i
            m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    return acc, l_i, m_i, n_valid_blocks


configs = [
    triton.Config({'BLOCK_M': BM, }, num_stages=s, num_warps=w) \
    # for BM in [64, 128] \
    # for BN in [32, 64] \
    # for s in ([1] if is_hip() else [3, 4, 7]) \
    # for w in [4, 8] \
    for BM in [64, ] \
    for s in ([1] if is_hip() else [3, ]) \
    for w in [4, ] \
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    if BLOCK_M * 32 < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit()
def _attn_fwd(Q, K, V, seq_info, seq_idx, sm_scale, M, N_VALID_BLOCKS, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_SeqInfoz, stride_SeqInfoh, stride_SeqInfon, stride_SeqInfonopt,  #
              stride_SeqIdxz, stride_SeqIdxh, stride_SeqIdxn,  #
              Z, H, N_CTX,  #
              N_CTX_NAtS,
              LOCAL_SEQ_MAX_LEN: tl.constexpr,
              N_REP: tl.constexpr,
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              CHUNK_NAtS: tl.constexpr,  #
              N_NAtS_BLOCK_PER_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    off_h_kv = off_h // N_REP
    kv_offset = off_z.to(tl.int64) * stride_kz + off_h_kv.to(tl.int64) * stride_kh
    seq_idx_offset = off_z.to(tl.int64) * stride_SeqIdxz + off_h_kv.to(tl.int64) * stride_SeqIdxh
    end_seq_info_offset = off_z.to(tl.int64) * stride_SeqInfoz + off_h_kv.to(tl.int64) * stride_SeqInfoh

    # how many

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )

    SeqsIdx_block_ptr = tl.make_block_ptr(
        base=seq_idx + seq_idx_offset,
        shape=(N_CTX_NAtS,),
        strides=(stride_SeqIdxn,),
        offsets=(0,),
        block_shape=(N_NAtS_BLOCK_PER_N,),
        order=(0,)
    )

    Seqinfo_block_ptr = tl.make_block_ptr(
        base=seq_info + end_seq_info_offset + 2,  # only the last element need to
        shape=(N_CTX_NAtS,),
        strides=(stride_SeqInfon * stride_SeqInfonopt,),
        offsets=(0,),  # only the last element need to be recorded
        block_shape=(N_NAtS_BLOCK_PER_N,),
        order=(0,)
    )

    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    n_valid_blocks = tl.zeros([BLOCK_M], dtype=tl.int32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(1,), padding_option='zero')
    # stage 1: off-band

    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    # This value indicates how many BLOCK_N
    if STAGE & 1:
        acc, l_i, m_i, n_valid_blocks = _attn_fwd_inner(acc, l_i, m_i, n_valid_blocks, q, K_block_ptr, V_block_ptr,
                                                        Seqinfo_block_ptr, SeqsIdx_block_ptr,  #
                                                        start_m, qk_scale,  #
                                                        BLOCK_M, HEAD_DIM, BLOCK_N, CHUNK_NAtS,  #
                                                        LOCAL_SEQ_MAX_LEN,
                                                        4 - STAGE, offs_m, offs_n, N_CTX,
                                                        N_NAtS_BLOCK_PER_N,
                                                        V.dtype.element_ty == tl.float8e5,
                                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i, n_valid_blocks = _attn_fwd_inner(acc, l_i, m_i, n_valid_blocks, q, K_block_ptr, V_block_ptr,
                                                        Seqinfo_block_ptr, SeqsIdx_block_ptr,  #
                                                        start_m, qk_scale,  #
                                                        BLOCK_M, HEAD_DIM, BLOCK_N, CHUNK_NAtS,  #
                                                        LOCAL_SEQ_MAX_LEN,
                                                        2, offs_m, offs_n, N_CTX,
                                                        N_NAtS_BLOCK_PER_N,
                                                        V.dtype.element_ty == tl.float8e5  #
                                                        )
    # epilogue
    m_i += tl.math.log2(l_i)

    acc = acc / l_i[:, None]
    # m_ptrs = M + off_hz * N_CTX + offs_m
    # n_valid_blocks_ptr = N_VALID_BLOCKS + off_hz * N_CTX + offs_m
    m_msks = offs_m < N_CTX
    tl.store(M + off_hz * N_CTX + offs_m, m_i, mask=m_msks)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(1,), )
    # TODO This value is implemented to estimate the sum exp values from each row during backward pass
    #  we estimate the mean row sum with (m_i / n_valid_blocks * n_actual_rows)
    # TODO check if this is a valid solution !!!
    tl.store(N_VALID_BLOCKS + off_hz * N_CTX + offs_m, n_valid_blocks, mask=m_msks)


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    msk_io = off_m < N_CTX
    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :], other=0, mask=msk_io[:, None]
    )
    do = tl.load(
        DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :], other=0, mask=msk_io[:, None]
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta, mask=msk_io)


# The main inner-loop logic for computing dK and dV.
# meanwhile, we also compute the gradients for the end seqs information
@triton.jit
def _attn_bwd_dkdv(dk, dv, d_seq_info0, d_seq_info1, d_seq_info2, d_seqinfo_collected,
                   is_global_token, is_local_token, fill_indices_scaled, seq_idx_lower_scaled, seq_idx_upper_scaled,
                   Q, k, v,
                   sm_scale,  #
                   DO,  #
                   M, D, N_VALID_COLUMNS,
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, LOCAL_SEQ_MAX_LEN, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   CHUNK_NAtS: tl.constexpr,
                   HEAD_DIM: tl.constexpr,  #
                   N_NAtS_BLOCK_PER_N1: tl.constexpr,
                   OnChipChunkSize: tl.constexpr,
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK_CASUAL: tl.constexpr,
                   OnlyComputedSeqInfo: tl.constexpr=False,
                   ):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d

    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1

    end_n_offset = tl.arange(0, N_NAtS_BLOCK_PER_N1) * OnChipChunkSize
    sw_token_end_idx = start_n + LOCAL_SEQ_MAX_LEN + end_n_offset + CHUNK_NAtS - 1
    for blk_idx in range(num_steps):

        offs_m = curr_m + tl.arange(0, BLOCK_M1)

        msk_m = offs_m < N_CTX
        msk_localT_base = tl.broadcast_to(fill_indices_scaled[:, None], (N_NAtS_BLOCK_PER_N1, BLOCK_M1)) > offs_m[None, :]

        if OnlyComputedSeqInfo:
            msk_swT = True
            mskT = True
            if N_NAtS_BLOCK_PER_N1 > 1 or CHUNK_NAtS > 1:
                msk_localT = tl.reshape(msk_localT_base, [N_NAtS_BLOCK_PER_N1, 1, BLOCK_M1])
                msk_localT = tl.broadcast_to(msk_localT, [N_NAtS_BLOCK_PER_N1, OnChipChunkSize, BLOCK_M1])
                msk_localT = tl.reshape(msk_localT, [BLOCK_N1, BLOCK_M1])
            else:
                msk_localT = msk_localT_base
        else:
            msk_swT = offs_m[None, :, ] <= sw_token_end_idx[:, None]
            if N_NAtS_BLOCK_PER_N1 == 1 or CHUNK_NAtS == 1:
                msk_local_swT = tl.where(is_local_token[:, None], msk_localT_base, msk_swT)
                mskT = tl.where(is_global_token[:, None], 1,  msk_local_swT)
                msk_localT = msk_localT_base
            else:
                msk_localT = tl.reshape(msk_localT_base, [N_NAtS_BLOCK_PER_N1, 1, BLOCK_M1])
                msk_swT = tl.reshape(msk_swT, [N_NAtS_BLOCK_PER_N1, 1, BLOCK_M1])

                msk_localsw_T = tl.where(is_local_token[:, None, None], msk_localT, msk_swT)
                mskT = tl.where(is_global_token[:, None, None], 1, msk_localsw_T)
                mskT = tl.broadcast_to(mskT, [N_NAtS_BLOCK_PER_N1, OnChipChunkSize, BLOCK_M1])
                mskT = tl.reshape(mskT, [BLOCK_N1, BLOCK_M1])

                msk_localT = tl.broadcast_to(msk_localT, [N_NAtS_BLOCK_PER_N1, OnChipChunkSize, BLOCK_M1])
                msk_localT = tl.reshape(msk_localT, [BLOCK_N1, BLOCK_M1])

                msk_swT = tl.broadcast_to(msk_swT, [N_NAtS_BLOCK_PER_N1, OnChipChunkSize, BLOCK_M1])
                msk_swT = tl.reshape(msk_swT, [BLOCK_N1, BLOCK_M1])

        do = tl.load(do_ptrs, mask=msk_m[:, None], other=0)
        Di = tl.load(D + offs_m, mask=msk_m, other=0)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)

        dpt_di = dpT - Di[None, :]
        qT = tl.load(qT_ptrs, mask=msk_m[None, :], other=0)
        # Load m before computing qk to reduce pipeline stall.

        m = tl.load(M + offs_m, mask=msk_m, other=torch.inf)
        n_valid_columns = tl.load(N_VALID_COLUMNS + offs_m, mask=msk_m, other=1)

        qkT = tl.dot(k, qT)

        pT_nomsk = tl.math.exp2(qkT - m[None, :])
        pT_nomsk = tl.clamp(pT_nomsk, 0., 1.)

        if MASK_CASUAL:
            msk_casualT = (offs_m[None, :] >= offs_n[:, None])
            pT_nomsk = tl.where(msk_casualT, pT_nomsk, 0)

        #pT_nomsk_scaled = pT_nomsk * (n_valid_columns / (offs_m + 1))[None, :]
        pT_nomsk_scaled = pT_nomsk
        dsT_scaled = (pT_nomsk_scaled * dpt_di).to(fp16_dtype)

        # now we could gather the gradients of the masks towards the end_seqs values.
        # the first is the set of the collected values
        msk_seq_values = offs_m[None, :] > fill_indices_scaled[:, None]
        global_token_not_in_next_seq_info = (offs_m[None, :] <= seq_idx_lower_scaled[:, None])
        msk_seq_collected = (offs_m[None, :] >= seq_idx_lower_scaled[:, None]) & (offs_m[None, :] < seq_idx_upper_scaled[:, None])

        if N_NAtS_BLOCK_PER_N1 == 1 or CHUNK_NAtS == 1:
            msk_seq_values_tt = is_global_token[:, None] & global_token_not_in_next_seq_info
            msk_seq_values0 = msk_seq_values & (~msk_seq_values_tt)
        else:
            #global_token_not_in_next_seq_info = tl.reshape(global_token_not_in_next_seq_info, [N_NAtS_BLOCK_PER_N1, 1, BLOCK_M1])
            msk_seq_values_tt = is_global_token[:, None] & global_token_not_in_next_seq_info
            msk_seq_values0 = msk_seq_values & (~msk_seq_values_tt)

            msk_seq_values0 = msk_seq_values0[:, None, :]

            msk_seq_values0 = tl.broadcast_to(msk_seq_values0, [N_NAtS_BLOCK_PER_N1, OnChipChunkSize, BLOCK_M1])
            msk_seq_values0 = tl.reshape(msk_seq_values0, [BLOCK_N1, BLOCK_M1])

            msk_seq_collected = tl.reshape(msk_seq_collected,  [N_NAtS_BLOCK_PER_N1, 1, BLOCK_M1])
            msk_seq_collected = tl.broadcast_to(msk_seq_collected, [N_NAtS_BLOCK_PER_N1, OnChipChunkSize, BLOCK_M1])
            msk_seq_collected = tl.reshape(msk_seq_collected, [BLOCK_N1, BLOCK_M1])

        d_seq_info0 += tl.sum(
            tl.where(msk_seq_values0, dsT_scaled, 0), 1
        )

        d_seqinfo_collected += tl.sum(
            tl.where(
                msk_seq_collected,
                dsT_scaled, 0
            ), 1
        )

        if tl.sum(msk_localT.to(tl.int32)) > 0:
            d_seq_info1 += tl.sum(
                tl.where(
                    msk_localT,
                    dsT_scaled, 0
                ), 1
            )

        if not OnlyComputedSeqInfo:
            pT = tl.where(mskT, pT_nomsk, 0.)

            # Compute dV.
            ppT = pT
            ppT = ppT.to(fp16_dtype)
            # D (= delta) is pre-divided by ds_scale.

            dsT = pT * dpt_di
            dsT = dsT.to(fp16_dtype)

            dv += tl.dot(ppT, do)
            dk += tl.dot(dsT, tl.trans(qT))

            d_seq_info2 += tl.sum(
                tl.where(
                    msk_swT,
                    dsT_scaled, 0
                ), 1
            )

        # we compute the gradients for masks
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok

    return dk, dv, d_seq_info0, d_seq_info1, d_seq_info2, d_seqinfo_collected


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq,
                 q, K, V, EndSeqsIDX, SeqInfo,  #
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 stride_end_seqs_idxn, stride_seqinfo_n,  #
                 H, N_CTX, LOCAL_SEQ_MAX_LEN,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 CHUNK_NAtS: tl.constexpr,  #
                 N_NAtS_CHUNK_PER_N2: tl.constexpr,
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr, LN2: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d

    msk_m = offs_m < N_CTX

    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m, mask=msk_m, other=0)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2

    end_n_offset = tl.arange(0, N_NAtS_CHUNK_PER_N2) * tl.minimum(CHUNK_NAtS, BLOCK_N2)

    for blk_idx in range(num_steps):
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        advance_nats = curr_n // CHUNK_NAtS + tl.arange(0, N_NAtS_CHUNK_PER_N2)

        end_seqs_idx_ptrs = EndSeqsIDX + advance_nats * stride_end_seqs_idxn
        sliding_window_ptrs = SeqInfo + advance_nats * stride_seqinfo_n + 2

        fill_indices = tl.load(end_seqs_idx_ptrs)
        fill_indices_scaled = fill_indices * CHUNK_NAtS + CHUNK_NAtS

        is_sw_token = tl.load(sliding_window_ptrs).to(tl.int1)
        is_global_token = fill_indices == advance_nats

        has_valid_indices = fill_indices_scaled > start_m

        sw_token_end_idx = curr_n + LOCAL_SEQ_MAX_LEN + end_n_offset + CHUNK_NAtS - 1
        has_valid_sw = sw_token_end_idx >= start_m

        do_sw_compute = tl.sum((is_sw_token & has_valid_sw).to(tl.int32))
        do_local_compute = tl.sum(((~is_sw_token) & has_valid_indices).to(tl.int32))
        #do_local_compute = tl.sum(((1 - is_sw_token.to(tl.int32)).to(tl.int1) & has_valid_indices).to(tl.int32))

        do_compute = (tl.sum(is_global_token.to(tl.int32)) + do_sw_compute + do_local_compute).to(tl.int1)

        # if tl.sum(msk.to(tl.int32)) > 0:
        if do_compute:

            msk_local = tl.broadcast_to(fill_indices_scaled[None, :], (BLOCK_M2, N_NAtS_CHUNK_PER_N2)) > offs_m[:, None]
            msk_sw = offs_m[:, None] <= sw_token_end_idx[None, :]
            msk_casual = (offs_m[:, None] >= offs_n[None, :])

            if N_NAtS_CHUNK_PER_N2 == 1 or CHUNK_NAtS == 1:
                msk_locals = tl.where(is_sw_token, msk_sw, msk_local)
                msk = tl.where(is_global_token, msk_casual, msk_casual & msk_locals)
            else:
                #
                msk_local = tl.reshape(msk_local, [BLOCK_M2, N_NAtS_CHUNK_PER_N2, 1])
                msk_sw = tl.reshape(msk_sw, [BLOCK_M2, N_NAtS_CHUNK_PER_N2, 1])
                msk_casual = tl.reshape(msk_casual, [BLOCK_M2, N_NAtS_CHUNK_PER_N2, CHUNK_NAtS])

                msk_locals = tl.where(is_sw_token[None, :, None], msk_sw, msk_local)
                msk = tl.where(is_global_token[None, :, None], msk_casual, msk_casual & msk_locals)
                msk = tl.reshape(msk, [BLOCK_M2, BLOCK_N2])

            msk_n = offs_n < N_CTX

            kT = tl.load(kT_ptrs, mask=msk_n[None, :], other=0)
            vT = tl.load(vT_ptrs, mask=msk_n[None, :], other=0)
            qk = tl.dot(q, kT)

            p = tl.where(msk, tl.math.exp2(qk - m), 0.)

            # Compute dP and dS.
            dp = tl.dot(do, vT).to(tl.float32)
            dp_di = dp - Di[:, None]
            ds = p * dp_di

            ds = ds.to(fp16_dtype)
            # Compute dQ.
            # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
            dq += tl.dot(ds, tl.trans(kT))
            # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok

    return dq


@triton.autotune([
    triton.Config({}, num_stages=s, num_warps=w) for s in ([1] if is_hip() else [3, 4, 5, 7]) for w in [4, 8]
],
    key=['N_CTX', 'HEAD_DIM'], )
@triton.jit
def _attn_bwd(Q, K, V, SeqInfo, EndSeqsIdx, EndSeqsUpper,
              sm_scale,  #
              DO,  #
              DQ, DK, DV, DSeqInfo, DCollectedValues,  #
              M, D, N_VALID_COLUMNS,
              # shared by Q/K/V/DO.
              stride_qz, stride_qh, stride_tok, stride_qd,  #
              stride_kz, stride_kh, stride_kn, stride_kd,
              stride_seqinfo_z, stride_seqinfo_h, stride_seqinfo_n, stride_seqinfo_nopt,
              stride_esq_z, stride_esq_h, stride_esq_n,
              stride_dseqinfo_z, stride_dseqinfo_h, stride_dseqinfo_n, stride_dseqinfo_nopt,
              stride_dseqcol_z, stride_dseqcol_h, stride_dseqcol_n,
              H, N_CTX, N_CTX_compressed,
              LOCAL_SEQ_MAX_LEN: tl.constexpr, n_rep: tl.constexpr,  #
              sparse_regularized_value: tl.constexpr,
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              CHUNK_NAtS: tl.constexpr,  #
              N_NAtS_BLOCK_PER_N1: tl.constexpr,
              N_NAtS_CHUNK_PER_MASK_N2: tl.constexpr,
              N_NAtS_CHUNK_PER_N2: tl.constexpr,
              OnChipChunkSize: tl.constexpr,
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)

    off_h = bhid % H
    off_z = bhid // H
    adj = (stride_qh * off_h + stride_qz * off_z).to(tl.int64)

    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    N_VALID_COLUMNS += off_chz
    D += off_chz

    off_h_kv = off_h // n_rep
    adj_kv = (stride_kh * off_h_kv + stride_kz * off_z).to(tl.int64)
    K += adj_kv
    V += adj_kv

    adj_end_seq = (stride_esq_h * off_h_kv + stride_esq_z * off_z).to(tl.int64)
    EndSeqsIdx += adj_end_seq
    EndSeqsUpper += adj_end_seq

    adj_seqinfo = (stride_seqinfo_z * off_z + stride_seqinfo_h * off_h_kv).to(tl.int64)
    SeqInfo += adj_seqinfo

    adj_end_seqinfo = (stride_dseqinfo_z * off_z + stride_dseqinfo_h * off_h).to(tl.int64)
    DSeqInfo += adj_end_seqinfo

    adj_dseqcol = (stride_dseqcol_z * off_z + stride_dseqcol_h * off_h).to(tl.int64)
    DCollectedValues += adj_dseqcol

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)
    #
    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    d_seq_info0 = tl.zeros([BLOCK_N1], dtype=tl.float32)
    d_seq_info1 = tl.zeros([BLOCK_N1], dtype=tl.float32)
    d_seq_info2 = tl.zeros([BLOCK_N1], dtype=tl.float32)
    d_collected_seq_values = tl.zeros([BLOCK_N1], dtype=tl.float32)

    # for conv gradients
    # compute offsets for w

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kd)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_kd)

    offset_nats = start_n // CHUNK_NAtS + tl.arange(0, N_NAtS_BLOCK_PER_N1)
    fill_indices = tl.load(EndSeqsIdx + offset_nats * stride_esq_n)
    seq_idx_lower = tl.load(
        EndSeqsIdx + offset_nats * stride_esq_n + 1, mask=offset_nats + 1 < N_CTX_compressed, other=N_CTX_compressed + 1
    )
    seq_idx_upper = tl.load(EndSeqsUpper + offset_nats * stride_esq_n)

    is_global_token = (fill_indices == offset_nats)
    # we only need the information on whether the end_seq_value is a pure local seq or not
    is_local_token = tl.load(SeqInfo + offset_nats * stride_seqinfo_n * stride_seqinfo_nopt + 1).to(tl.int1)

    fill_indices_scaled = fill_indices * CHUNK_NAtS + CHUNK_NAtS
    seq_idx_lower_scaled = seq_idx_lower * CHUNK_NAtS + CHUNK_NAtS
    seq_idx_upper_scaled = seq_idx_upper * CHUNK_NAtS + CHUNK_NAtS

    last_idx_sw = start_n + tl.arange(0, N_NAtS_BLOCK_PER_N1) * CHUNK_NAtS + LOCAL_SEQ_MAX_LEN + CHUNK_NAtS
    last_idx = tl.where(is_global_token, N_CTX, last_idx_sw)
    # We want to compute until the SW tokens anyway
    last_idx_sw_max = start_n + (N_NAtS_BLOCK_PER_N1 - 1) * CHUNK_NAtS + LOCAL_SEQ_MAX_LEN + CHUNK_NAtS
    last_idx_locals = tl.where(fill_indices_scaled >= last_idx_sw_max, fill_indices_scaled, last_idx_sw_max)

    last_idx = tl.max(tl.where(is_local_token, last_idx_locals, last_idx))

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv, d_seq_info0, d_seq_info1, d_seq_info2, d_collected_seq_values = _attn_bwd_dkdv(
        dk, dv, d_seq_info0, d_seq_info1, d_seq_info2, d_collected_seq_values,
        is_global_token, is_local_token, fill_indices_scaled, seq_idx_lower_scaled, seq_idx_upper_scaled,
        Q, k, v, sm_scale,  #
        DO,  #
        M, D, N_VALID_COLUMNS, #
        stride_tok, stride_kd,  #
        H, N_CTX, LOCAL_SEQ_MAX_LEN,  #
        MASK_BLOCK_M1, BLOCK_N1, CHUNK_NAtS, HEAD_DIM, N_NAtS_BLOCK_PER_N1, OnChipChunkSize, #
        start_n, start_m, num_steps,  #
        MASK_CASUAL=True  #
    )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = tl.cdiv(last_idx - start_m, BLOCK_M1)

    # Compute dK and dV for non-masked blocks.
    dk, dv, d_seq_info0, d_seq_info1, d_seq_info2, d_collected_seq_values = _attn_bwd_dkdv(  #
        dk, dv, d_seq_info0, d_seq_info1, d_seq_info2, d_collected_seq_values,
        #
        is_global_token, is_local_token, fill_indices_scaled, seq_idx_lower_scaled, seq_idx_upper_scaled,
        Q, k, v, sm_scale,  #
        DO,  #
        M, D, N_VALID_COLUMNS, #
        stride_tok, stride_kd,  #
        H, N_CTX, LOCAL_SEQ_MAX_LEN,  #
        BLOCK_M1, BLOCK_N1, CHUNK_NAtS, HEAD_DIM, N_NAtS_BLOCK_PER_N1, OnChipChunkSize, #
        start_n, start_m, num_steps,  #
        MASK_CASUAL=False  #
    )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = tl.cdiv(N_CTX - start_m, BLOCK_M1)

    # STAGE3, we only compute the mask values

    dk, dv, d_seq_info0, d_seq_info1, d_seq_info2, d_collected_seq_values = _attn_bwd_dkdv(  #
        dk, dv, d_seq_info0, d_seq_info1, d_seq_info2, d_collected_seq_values,
        #
        is_global_token, is_local_token, fill_indices_scaled, seq_idx_lower_scaled, seq_idx_upper_scaled,
        Q, k, v, sm_scale,  #
        DO,  #
        M, D, N_VALID_COLUMNS, #
        stride_tok, stride_kd,  #
        H, N_CTX, LOCAL_SEQ_MAX_LEN,  #
        BLOCK_M1, BLOCK_N1, CHUNK_NAtS, HEAD_DIM, N_NAtS_BLOCK_PER_N1, OnChipChunkSize,  #
        start_n, start_m, num_steps,  #
        MASK_CASUAL=False,  #
        OnlyComputedSeqInfo=True
    )

    d_seq_info0 = tl.sum(tl.reshape(d_seq_info0, [N_NAtS_BLOCK_PER_N1, OnChipChunkSize]) / CHUNK_NAtS, -1)
    d_seq_info1 = tl.sum(tl.reshape(d_seq_info1, [N_NAtS_BLOCK_PER_N1, OnChipChunkSize]) / CHUNK_NAtS, -1)
    d_seq_info2 = tl.sum(tl.reshape(d_seq_info2, [N_NAtS_BLOCK_PER_N1, OnChipChunkSize]) / CHUNK_NAtS, -1)
    d_collected_seq_values = tl.sum(tl.reshape(d_collected_seq_values, [N_NAtS_BLOCK_PER_N1, OnChipChunkSize]) / CHUNK_NAtS, -1)

    d_seq_info0 += tl.where(is_global_token, N_CTX_compressed - fill_indices, 0.) / N_CTX_compressed * sparse_regularized_value
    d_seq_info1 += tl.where(is_local_token, fill_indices - offset_nats, 0.) / N_CTX_compressed * sparse_regularized_value

    offset_dseqinfo = start_n // OnChipChunkSize + tl.arange(0, N_NAtS_BLOCK_PER_N1)
    DSeqInfo = DSeqInfo + offset_dseqinfo * stride_dseqinfo_n * stride_dseqinfo_nopt
    tl.store(DSeqInfo, d_seq_info0)
    tl.store(DSeqInfo + 1, d_seq_info1)
    tl.store(DSeqInfo + 2, d_seq_info2)

    tl.store(DCollectedValues + offset_dseqinfo * stride_dseqcol_n, d_collected_seq_values)

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_kd
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_kd
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    # This does not pass the semantic check from triton, need to find how to solve this....
    # N_NAtS_CHUNK_PER_MASK_N2: tl.constexpr = max(MASK_BLOCK_N2 // CHUNK_NAtS, 1) #tl.cdiv(MASK_BLOCK_N2, CHUNK_NAtS)
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_qd)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_qd)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(dq,
                      q, K, V, EndSeqsIdx, SeqInfo,  #
                      do, m, D,  #
                      stride_tok, stride_qd,  #
                      stride_esq_n, stride_seqinfo_n * stride_seqinfo_nopt,  #
                      H, N_CTX, LOCAL_SEQ_MAX_LEN,  #
                      BLOCK_M2,
                      MASK_BLOCK_N2,
                      CHUNK_NAtS,
                      N_NAtS_CHUNK_PER_MASK_N2,
                      HEAD_DIM,  #
                      start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
                      LN2=LN2, MASK=True  #
                      )
    end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(dq,
                      q, K, V, EndSeqsIdx, SeqInfo,  #
                      do, m, D,  #
                      stride_tok, stride_qd,  #
                      stride_esq_n, stride_seqinfo_n * stride_seqinfo_nopt,  #
                      H, N_CTX, LOCAL_SEQ_MAX_LEN,  #
                      BLOCK_M2,
                      BLOCK_N2,
                      CHUNK_NAtS,
                      N_NAtS_CHUNK_PER_N2,
                      HEAD_DIM,  #
                      start_m, end_n - num_steps * BLOCK_N2, num_steps,  #
                      LN2=LN2, MASK=False  #
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_qd
    dq *= LN2
    tl.store(dq_ptrs, dq)


@triton.jit
def shift_min(v0, i0, j0, v1, i1, j1):
    gt = (i1 < i0)
    # gt = (v0 != v1)
    return tl.where(gt, tl.where(j1, v1, i0), v0), i1, tl.where(j0, True, tl.where(gt, True, False))


@triton.autotune(
    [
        #triton.Config({'BLOCK_M': 4, }, num_stages=4, num_warps=8),
        #triton.Config({'BLOCK_M': 2, }, num_stages=4, num_warps=8),
        #triton.Config({'BLOCK_M': 1, }, num_stages=4, num_warps=8),

        #triton.Config({'BLOCK_M': 4, }, num_stages=2, num_warps=8),
        #triton.Config({'BLOCK_M': 2, }, num_stages=2, num_warps=8),
        #triton.Config({'BLOCK_M': 1, }, num_stages=2, num_warps=8),

        #triton.Config({'BLOCK_M': 4, }, num_stages=5, num_warps=8),
        #triton.Config({'BLOCK_M': 2, }, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 1, }, num_stages=5, num_warps=8),
    ],
    key=['BATCH', 'N_HEAD', 'N_CTX'],
)
@triton.jit
def _shift_end_seqs_idx(EndSeqsIDX,
                        EndSeqsLOWERIDX,
                        EndSeqsUpperIDX,
                        stride_xz, stride_xh, stride_xn,  #
                        BATCH, N_HEAD, N_CTX,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    This function takes the EndSeqsIDX as input and returns 2 shifted values:
    The first one is the last N_CTX - 1 values from EndSeqsIDX, the second one is shifted from EndSeqsIDX: each element
    in EndSeqsIDX will be replaced with the first item in the list that is larger than it:
    [0, 2, 2, 5, 5, 5, 6] will be replaced with
    [2, 5, 5, 6, 6, 6, 6]
    this is equivalent to the following function but with an inversed order:
    def shift_value(esi: torch.Tensor):
        out = torch.empty_like(esi)
        out[0] = 0
        for i in range(1, esi.shape[-1]):
            if esi[i] > esi[i - 1]:
                out[i] = esi[i - 1]
            else:
                out[i] = out[i - 1]
        return out
    """
    range_n = tl.arange(0, BLOCK_N) + 1
    range_m = tl.arange(0, BLOCK_M)
    pid = tl.program_id(axis=0) * BLOCK_M  # We use a 1D launch grid so axis is 0.

    off_z = pid // N_HEAD
    off_h = pid % N_HEAD
    zh_offset = off_z.to(tl.int64) * stride_xz + off_h.to(tl.int64) * stride_xh

    off_m = zh_offset + range_m[:, None] * stride_xh
    mask_n = range_n[None, :] < N_CTX
    mask_m = off_m < BATCH * N_HEAD * stride_xh

    # This allows us to move the data backward once
    lower = tl.load(
        EndSeqsIDX + off_m + range_n[None, :], mask=mask_m & mask_n,
        other=N_CTX - 1
    )

    upper, _, _ = tl.associative_scan(
        # (end_seqs_idx, tl.broadcast_to(range_n[None, :], (BLOCK_M, BLOCK_N)).to(end_seqs_idx.dtype)),
        (lower, lower, tl.broadcast_to(tl.zeros([BLOCK_N], dtype=tl.int1)[None, :], (BLOCK_M, BLOCK_N))),
        # (end_seqs_idx, tl.broadcast_to(tl.zeros_like(range_n)[None, :], (BLOCK_M, BLOCK_N)).to(end_seqs_idx.dtype)),
        # (end_seqs_idx, end_seqs_idx),
        axis=-1,
        combine_fn=shift_min,
        reverse=True
    )

    off_m_store = zh_offset + range_m[:, None] * stride_xh
    mask_m_store = off_m_store < BATCH * N_HEAD * stride_xh

    out_range = tl.arange(0, BLOCK_N)
    out_msk = (out_range < N_CTX) & mask_m_store

    tl.store(
        EndSeqsLOWERIDX + off_m_store + out_range[None, :], lower,
        mask=out_msk,
    )

    tl.store(
        EndSeqsUpperIDX + off_m_store + out_range[None, :], upper,
        mask=out_msk,
    )


class NAtSChunkAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, seq_info, end_seqs_idx, causal, sm_scale,
                nats_chunk_size: int = 32,
                n_rep: int = 1,
                sparse_regularized_value=0.,
                local_seq_max_length: int = 4,
                compress_on_q: bool = False,
                ):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)
        stage = 3
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        # this value is used to record the number of valid blocks for each row. Then we could directly estimate the sum
        # (denominators) of each softmax values to avoid 2 iteration to compute the exact denoinator
        n_valid_blocks = torch.zeros(M.shape, dtype=torch.long, device=q.device)

        block_n = 32
        _attn_fwd[grid](
            q, k, v, seq_info, end_seqs_idx, sm_scale, M, n_valid_blocks, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            seq_info.stride(0), seq_info.stride(1), seq_info.stride(2), seq_info.stride(3),  #
            end_seqs_idx.stride(0), end_seqs_idx.stride(1), end_seqs_idx.stride(2),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            N_CTX_NAtS=end_seqs_idx.shape[2],
            LOCAL_SEQ_MAX_LEN=local_seq_max_length,
            N_REP=n_rep,
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            CHUNK_NAtS=nats_chunk_size,
            BLOCK_N=block_n,
            N_NAtS_BLOCK_PER_N=math.ceil(block_n / nats_chunk_size),
            **extra_kern_args)
        ctx.save_for_backward(q, k, v, seq_info, end_seqs_idx, o, M, n_valid_blocks)
        ctx.grid = grid
        ctx.nats_chunk_size = nats_chunk_size
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.sparse_regularized_value = sparse_regularized_value
        ctx.local_seq_max_length = local_seq_max_length
        ctx.n_rep = n_rep
        ctx.compress_on_q = compress_on_q
        return o

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> Any:
        q, k, v, seq_info, end_seqs_idx, o, M, n_valid_blocks = ctx.saved_tensors
        # Since we transposed o in the attention forward pass, the memor format of do might be corrupted...
        do = do.contiguous()
        # assert do.is_contiguous()
        # assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        assert q.stride() == o.stride() == do.stride()
        n_rep = ctx.n_rep
        BATCH, N_HEAD, N_CTX, HEAD_DIM = q.shape

        N_HEAD_KV = end_seqs_idx.shape[1]

        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        nats_chunk_size = ctx.nats_chunk_size

        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        )

        dq = torch.empty_like(q)
        # for dk, dv, we need to have [B, H_K * n_rep, N_CTX, HEAD_DIM]
        dk = torch.empty_like(q)
        dv = torch.empty_like(q)

        local_seq_max_length = ctx.local_seq_max_length

        N_CTX_compressed = end_seqs_idx.shape[-1]

        if BLOCK_N1 >= nats_chunk_size:
            d_seq_info = torch.empty(
                [BATCH, N_HEAD, N_CTX_compressed, 3],
                device=seq_info.device, dtype=seq_info.dtype
            )
            d_seqinfo_collected = torch.empty(
                [BATCH, N_HEAD, N_CTX_compressed], device=seq_info.device, dtype=seq_info.dtype,
            )
            OnChipChunkSize = nats_chunk_size
        else:
            assert nats_chunk_size // BLOCK_N1 == 0
            n_ctx_dseqinfo = math.ceil(N_CTX / BLOCK_N1)
            d_seq_info = torch.empty(
                [BATCH, N_HEAD, n_ctx_dseqinfo, 3],
                device=seq_info.device, dtype=seq_info.dtype
            )
            d_seqinfo_collected = torch.empty(
                [BATCH, N_HEAD, n_ctx_dseqinfo], device=seq_info.device, dtype=seq_info.dtype,
            )
            OnChipChunkSize = BLOCK_N1

        endseqs_lower = torch.empty_like(end_seqs_idx)
        endseqs_upper = torch.empty_like(end_seqs_idx)

        grid = lambda args: (triton.cdiv(BATCH * N_HEAD_KV, args['BLOCK_M']),)
        # NOTE:
        #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
        #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
        #  - Don't forget to pass meta-parameters as keywords arguments.

        _shift_end_seqs_idx[grid](end_seqs_idx, endseqs_lower, endseqs_upper,
                                  end_seqs_idx.stride(0), end_seqs_idx.stride(1), end_seqs_idx.stride(2),
                                  BATCH, N_HEAD, N_CTX_compressed,
                                  BLOCK_N=triton.next_power_of_2(N_CTX_compressed))


        #"""

        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, arg_k, v, seq_info, end_seqs_idx, endseqs_upper,
            ctx.sm_scale,
            do, dq, dk, dv, d_seq_info, d_seqinfo_collected, #
            M, delta, n_valid_blocks,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            seq_info.stride(0), seq_info.stride(1), seq_info.stride(2), seq_info.stride(3),  #
            end_seqs_idx.stride(0), end_seqs_idx.stride(1), end_seqs_idx.stride(2),  #
            d_seq_info.stride(0), d_seq_info.stride(1), d_seq_info.stride(2), d_seq_info.stride(3),
            d_seqinfo_collected.stride(0), d_seqinfo_collected.stride(1), d_seqinfo_collected.stride(2),
            N_HEAD, N_CTX, N_CTX_compressed,
            local_seq_max_length, n_rep,  ctx.sparse_regularized_value, #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            CHUNK_NAtS=nats_chunk_size,
            N_NAtS_BLOCK_PER_N1=math.ceil(BLOCK_N1 / nats_chunk_size),
            N_NAtS_CHUNK_PER_N2=math.ceil(BLOCK_N2 / nats_chunk_size),
            N_NAtS_CHUNK_PER_MASK_N2=math.ceil((BLOCK_N2 // BLK_SLICE_FACTOR) / nats_chunk_size),
            OnChipChunkSize=OnChipChunkSize,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
        )

        #"""
        if BLOCK_N1 >= nats_chunk_size:
            d_seq_info = d_seq_info.view(BATCH, N_HEAD, N_CTX_compressed, -1, 3).sum(-2)
            d_seqinfo_collected = d_seqinfo_collected.view(BATCH, N_HEAD, N_CTX_compressed, -1, ).sum(-1)
        if n_rep > 1:
            dk = dk.view(BATCH, N_HEAD_KV, n_rep, N_CTX, HEAD_DIM).sum(2)
            dv = dv.view(BATCH, N_HEAD_KV, n_rep, N_CTX, HEAD_DIM).sum(2)

            d_seq_info = d_seq_info.view(BATCH, N_HEAD_KV, n_rep, N_CTX_compressed, 3).sum(2)

            d_seqinfo_collected = d_seqinfo_collected.view(BATCH, N_HEAD_KV, n_rep, N_CTX_compressed).sum(2)

        d_seq_info[:, :, :, 0] = d_seq_info[:, :, :, 0].scatter_add(-1, endseqs_lower, -d_seqinfo_collected)

        return dq, dk, dv, d_seq_info, None, None, None, None, None, None, None, None


def nats_chunk_attention(q, k, v, seq_info, end_seqs_idx, causal, sm_scale, nats_chunk_size, n_rep: int = 1,
                         sparse_regularized_value=0., local_seq_max_length: int = 4, compress_on_q: bool = True):
    """
    Here we compute the segment wise attention on-chip. This avoids the preservation of the attention masks and its
    gradients. Specially, we use end_seqs_idx to determine the form of the masks and pass the gradients information
    back to the end_seqs_hard. Hence, end_seqs_hard is not applied here during forward pass but it remains required
    during backward pass.
    TODO: check if putting end_seqs_hard into the forward pass is faster than the current solution!
    Args:
        q: torch.Tensor of shape [bsz, nheads, N_ctx, D_HEAD], attention query
        k: torch.Tensor of shape [bsz, nheads, N_ctx, D_HEAD], attention key
        v: torch.Tensor of shape [bsz, nheads, N_ctx, D_HEAD], attention values
        seq_info: torch.Tensor of shape [bsz, nheads, N_ctx, N_opts], this value indicates the states of each token
            currently,
            1. if the first option is selected, this token will be preserved until the end.
            2. if the second option is selected, this token will be preserved until the next token whose end_seqs_info
               is selected as the first token
            3. if the third option is selected, this token will be preserved for the next few tokens (4 or 8)
        end_seqs_idx: torch.Tensor of shape [bsz, nheads, N_ctx], this is the reversed cummax indices of
            end_seqs_info that select the first item and is used to construct the attetion maps
        sm_scale: scales for computing attention maps, should always be `sm_scale = 1 / math.sqrt(D_HEAD)`
        nats_chunk_size: int, the block size for computing NAtS Attention
        n_rep: number of replications for KV and end_seqs
        sparse_regularized_value: a regularization term controlling the
        local_seq_max_length: int, the maximal lenght of the local sequences

    Returns:
        o: torch.Tensor of shape [bsz, nheads, N_ctx, D_HEAD], attention output

    """
    return NAtSChunkAttention.apply(q, k, v, seq_info, end_seqs_idx, causal, sm_scale, nats_chunk_size, n_rep,
                                    sparse_regularized_value, local_seq_max_length, compress_on_q)


def attention_torch(q, k, v, mask):
    import math
    dtype = torch.float16 if fp_16_type == 'float16' else torch.bfloat16

    b, h, nctx, d = q.shape
    # qk = qk / math.sqrt(d * h) + mask

    # scores = torch.matmul(q, k.transpose(2, 3)) * (1.0 / math.sqrt(d)) + mask
    # qk.retain_grad()
    # scores = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(q)

    qk1 = torch.matmul(q, torch.transpose(k, -1, -2)) * (1.0 / math.sqrt(d))
    qk2 = qk1 - qk1.max(-1, keepdim=True).values
    qk2.retain_grad()
    qk2_exp_ = torch.exp(qk2)
    qk2_exp_.retain_grad()
    qk2_exp = qk2_exp_ * mask
    qk2_exp.retain_grad()
    scores = (qk2_exp / (qk2_exp.sum(-1, keepdim=True))).to(dtype=dtype)

    o = torch.matmul(scores, v)  # (bs, n_local_heads, seqlen, head_dim)

    # p = torch.softmax(qk, dim=-1)
    # p.retain_grad()
    # o = p @ v

    # from torch.nn.functional import scaled_dot_product_attention
    # res2 = scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=None)
    return o, qk2, qk2_exp, scores


def attention_torch_add(q, k, v, mask):
    import math

    b, h, nctx, d = q.shape
    # qk = qk / math.sqrt(d * h) + mask

    # scores = torch.matmul(q, k.transpose(2, 3)) * (1.0 / math.sqrt(d)) + mask
    # qk.retain_grad()
    # scores = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(q)

    qk1 = torch.matmul(q, torch.transpose(k, -1, -2)) * (1.0 / math.sqrt(d))
    qk2 = qk1 - qk1.max(-1, keepdim=True).values + mask
    qk2.retain_grad()
    qk2_exp_ = torch.exp(qk2)
    qk2_exp_.retain_grad()
    qk2_exp = qk2_exp_
    qk2_exp.retain_grad()
    scores = (qk2_exp / (qk2_exp.sum(-1, keepdim=True))).half()

    o = torch.matmul(scores, v)  # (bs, n_local_heads, seqlen, head_dim)

    # p = torch.softmax(qk, dim=-1)
    # p.retain_grad()
    # o = p @ v

    # from torch.nn.functional import scaled_dot_product_attention
    # res2 = scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=None)
    return o, qk2, qk2_exp, scores


def test_flash_atn():
    # TODO move this to tests!
    torch.manual_seed(0)
    from nats.components.masks.triton import construct_soft_mask, cummax_reverse
    import math
    BATCH = 2
    H = 4
    N_CTX = 1024
    D_HEAD = 64
    nats_chunk_size = 1
    dtype = torch.float16 if fp_16_type == 'float16' else torch.bfloat16
    device = torch.device("cuda")
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)

    base_value = torch.arange(N_CTX, device=device, dtype=dtype).view(1, 1, -1, 1) / N_CTX
    # q = torch.ones_like(q, dtype=dtype, device=device, requires_grad=True) *base_value
    # k = torch.ones_like(k, dtype=dtype, device=device, requires_grad=True) *base_value
    # v = torch.ones_like(v, dtype=dtype, device=device, requires_grad=True) *base_value

    n_ctx_compressed = N_CTX // nats_chunk_size
    sm_scale = 1 / math.sqrt(D_HEAD)
    sliding_window_size = 16
    from torch.nn import Transformer
    mask = Transformer.generate_square_subsequent_mask(N_CTX, device=device, dtype=dtype)
    mask_ = torch.where(mask == 0, 1., 0.)

    gumbels = (
        -torch.empty([BATCH, H, n_ctx_compressed, 3], memory_format=torch.legacy_contiguous_format).exponential_().log()
    ).cuda()

    # gumbels[:,:,:,0] = -torch.inf
    # gumbels[:,:,:,1] = -torch.inf
    # gumbels[:,:,:,2] = -torch.inf

    y_soft = gumbels.softmax(-1)
    index = y_soft.max(-1, keepdim=True)[1]
    index[:, :, -1, :] = 0

    y_hard = torch.zeros_like(gumbels, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
    end_seqs_hard = y_hard - y_soft.detach() + y_soft

    end_seqs_hard = end_seqs_hard.cuda().to(dtype=dtype)

    if torch.all(end_seqs_hard[:, :, :, 0] == 0):
        end_seqs_indices = torch.ones([BATCH, H, n_ctx_compressed], dtype=torch.int64, ).cuda() * (N_CTX - 1)
    else:
        end_seqs_indices = n_ctx_compressed - 1 - torch.flip(
            torch.cummax(torch.flip(end_seqs_hard[:, :, :, 0], (-1,)), -1)[1], (-1,))

    import copy

    q0 = copy.deepcopy(q.detach())
    k0 = copy.deepcopy(k.detach())
    v0 = copy.deepcopy(v.detach())
    q0 = torch.nn.Parameter(q0, requires_grad=True)
    k0 = torch.nn.Parameter(k0, requires_grad=True)
    v0 = torch.nn.Parameter(v0, requires_grad=True)
    end_seqs_hard0 = torch.nn.Parameter(end_seqs_hard, requires_grad=True)

    res1 = nats_chunk_attention(q0, k0, v0, end_seqs_hard0, end_seqs_indices, True, sm_scale, nats_chunk_size, 1, 1e-8,
                                sliding_window_size)

    loss = (res1 ** 2).sum()
    loss.backward()

    mks_local = construct_soft_mask(end_seqs_indices, end_seqs_hard.float()[:, :, :, 0].contiguous(),
                                    torch.gather(end_seqs_hard.float()[:, :, :, 0].contiguous(), -1, end_seqs_indices),
                                    0)
    mks_local = torch.where(mks_local == 0., 1., 0.)
    mks_local = mks_local.repeat_interleave(nats_chunk_size, 2).repeat_interleave(nats_chunk_size, 3)

    # local_msks = ((torch.arange(N_CTX) + sliding_window_size + 1).unsqueeze(0) > torch.arange(N_CTX).unsqueeze(1)).float().cuda()
    local_msks = ((torch.arange(n_ctx_compressed) + sliding_window_size // nats_chunk_size + 1).unsqueeze(
        0) > torch.arange(n_ctx_compressed).unsqueeze(1))
    local_msks = local_msks.float().cuda()
    local_msks = local_msks.repeat_interleave(nats_chunk_size, -1).repeat_interleave(nats_chunk_size, -2)
    msk_sw = end_seqs_hard[:, :, :, -1].unsqueeze(-2) == 1.
    msk_sw = msk_sw.repeat_interleave(nats_chunk_size, -1)
    msk1 = torch.where(msk_sw, local_msks.view(1, 1, N_CTX, N_CTX), mks_local)

    # msk_local = ((torch.arange(N_CTX) + 4).unsqueeze(0) >= (torch.arange(N_CTX)).unsqueeze(1)).float().cuda().view(1, 1,
    #                                                                                                               N_CTX,
    #                                                                                                               N_CTX)
    # msk1 = torch.where((end_seqs_hard == 1.).unsqueeze(-2), msk1, msk1 * msk_local)
    from nats.models.transformer.triton.flash_attention_origin import _attention2
    mask1 = torch.nn.Parameter(data=msk1.data, requires_grad=True)
    # mask1 = torch.ones_like(mask1).tril_()

    q1 = copy.deepcopy(q.detach())
    k1 = copy.deepcopy(k.detach())
    v1 = copy.deepcopy(v.detach())
    q1 = torch.nn.Parameter(q1, requires_grad=True)
    k1 = torch.nn.Parameter(k1, requires_grad=True)
    v1 = torch.nn.Parameter(v1, requires_grad=True)

    res2, qk, qk2_exp, p = attention_torch(q1, k1, v1, mask1 * mask_)
    #  mask & (offs_m[:, None] <= start_n + offs_n[None, :] + LOCAL_SEQ_MAX_LEN)

    loss2 = (res2 ** 2).sum()
    loss2.backward()
    # end_seq_soft_grad, end_seq_collected_grad = compute_grad_soft_mask(end_seqs_indices, end_seqs_hard, mask1.grad)

    print(f'diff res max:{(res1 - res2).max()}')
    print(f'diff res min:{(res1 - res2).min()}')
    print(f'diff q0 grad max:{(q0.grad - q1.grad).max()}')
    print(f'diff q1 grad min:{(q0.grad - q1.grad).max()}')
    print(f'diff k0 grad max:{(k0.grad - k1.grad).max()}')
    print(f'diff k1 grad min:{(k0.grad - k1.grad).max()}')
    print(f'diff v0 grad max:{(v0.grad - v1.grad).max()}')
    print(f'diff v1 grad min:{(v0.grad - v1.grad).max()}')

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    test_flash_atn()
    # bench_flash_attention(BATCH, N_HEADS, N_CTX, D_HEAD, True, 'bwd', 'triton', False)
    # bench_flash_attention.run(save_path=".", print_data=True)
# """
