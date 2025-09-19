# TODO this is the on-chip version of flash attention
#  instead of receiving the exact mask it only receives the end_seqs_hard and idx_end_seqs information
#  all the masks related values, including forward and backward, will be computed on chip. Therefore, there is no
#  need for us to preserve the N x N masks during backward pass
from typing import Any
import torch

import triton
import triton.language as tl

from nats.utils import check_fp16_dtype

fp_16_type = check_fp16_dtype()

fp16_dtype = tl.float16 if fp_16_type == 'float16' else tl.bfloat16


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _attn_fwd_inner(acc, l_i, l_i_nomsk, m_i, q,  #
                    K_block_ptr, V_block_ptr, Seqinfo_block_ptr, SeqsIdx_block_ptr, #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    LOCAL_SEQ_MAX_LEN: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
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
    fill_indices_ptr = tl.advance(SeqsIdx_block_ptr, lo)
    sliding_window_tokens_ptr = tl.advance(Seqinfo_block_ptr, lo)

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        offs_n_ = start_n + offs_n
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)

        fill_indices = tl.load(fill_indices_ptr)
        sliding_window_tokens = tl.load(sliding_window_tokens_ptr)
        # we first check if the mask is on the upper part of the attention maps, then we check if the mask
        # belongs to the local attention or full attention parts (identified by fill_indices)
        # finally, the global tokens must be preserved
        mask_casual = (offs_m[:, None] >= offs_n_[None, :])

        if STAGE == 2:
            qk = qk * qk_scale + tl.where(mask_casual, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        mask = (fill_indices[None, :] >= offs_m[:, None])
        # mask = mask & (offs_m[:, None] <= start_n + offs_n[None, :] + LOCAL_SEQ_MAX_LEN)
        mask = tl.where(sliding_window_tokens[None, :],
                        (offs_m[:, None] <= (offs_n_[None, :] + LOCAL_SEQ_MAX_LEN)),
                        mask
                        )

        mask = mask | tl.broadcast_to(fill_indices == offs_n_[None, :], (BLOCK_M, BLOCK_N))
        mask = mask & mask_casual

        p_nomsk = tl.math.exp2(qk)

        p = tl.where(mask, p_nomsk, 0)

        l_ij = tl.sum(p, 1)
        l_ij_nomsk = tl.sum(p_nomsk, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        l_i_nomsk = l_i_nomsk * alpha + l_ij_nomsk
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(fp16_dtype)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        fill_indices_ptr = tl.advance(fill_indices_ptr, (BLOCK_N,))

        sliding_window_tokens_ptr = tl.advance(sliding_window_tokens_ptr, (BLOCK_N,))

    return acc, l_i, l_i_nomsk, m_i


configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128] \
    for BN in [32, 64] \
    for s in ([1] if is_hip() else [3, 4, 7]) \
    for w in [4, 8] \
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit()
def _attn_fwd(Q, K, V, seq_info, seq_idx, sm_scale, M, M_nomsk, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_SeqInfoz, stride_SeqInfoh, stride_SeqInfon, stride_SeqInfonopt,  #
              stride_SeqIdxz, stride_SeqIdxh, stride_SeqIdxn,  #
              Z, H, N_CTX,  #
              LOCAL_SEQ_MAX_LEN: tl.constexpr,
              N_REP: tl.constexpr,
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
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
    end_seq_info_pffset = off_z.to(tl.int64) * stride_SeqInfoz + off_h_kv.to(tl.int64) * stride_SeqInfoh

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
        shape=(N_CTX,),
        strides=(stride_SeqIdxn,),
        offsets=(0,),
        block_shape=(BLOCK_N,),
        order=(0,)
    )

    Seqinfo_block_ptr = tl.make_block_ptr(
        base=seq_info + end_seq_info_pffset + 2,  # only the last element need to
        shape=(N_CTX,),
        strides=(stride_SeqInfon * stride_SeqInfonopt,),
        offsets=(0,),  # only the last element need to be recorded
        block_shape=(BLOCK_N,),
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
    l_i_nomsk = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, l_i_nomsk, m_i = _attn_fwd_inner(acc, l_i, l_i_nomsk, m_i, q, K_block_ptr, V_block_ptr,
                                                   Seqinfo_block_ptr, SeqsIdx_block_ptr, #
                                                   start_m, qk_scale,  #
                                                   BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                                   LOCAL_SEQ_MAX_LEN,
                                                   4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5
                                                   )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, l_i_nomsk, m_i = _attn_fwd_inner(acc, l_i, l_i_nomsk, m_i, q, K_block_ptr, V_block_ptr,
                                                   Seqinfo_block_ptr, SeqsIdx_block_ptr, #
                                                   start_m, qk_scale,  #
                                                   BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                                   LOCAL_SEQ_MAX_LEN,
                                                   2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                                   )
    # epilogue
    m_i_nomsk = m_i + tl.math.log2(l_i_nomsk)
    m_i += tl.math.log2(l_i)

    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(M_nomsk + off_hz * N_CTX + offs_m, m_i_nomsk)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


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
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
# meanwhile, we also compute the gradients for the end seqs information
@triton.jit
def _attn_bwd_dkdv(dk, dv,
                   d_end_seq_values0, d_end_seq_values1, d_end_seq_values2, d_collected_values, local_tokens,
                   DConvValues,  #
                   conv_first_m, conv_last_m, conv_store_idx,
                   conv_kernel_size_w: tl.constexpr, conv_kernel_size_h: tl.constexpr,  #
                   Q, k, v,
                   fill_indices, d_endseqs_lower, d_endseqs_upper,
                   sm_scale,  #
                   DO,  #
                   M, M_nomsk, D, DSeqInfo,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, LOCAL_SEQ_MAX_LEN, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d

    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1

    # See following on how h_offsetT can be computed
    h_offsetT = tl.arange(0, BLOCK_M1)[None, None, :] - 1 - tl.arange(0, conv_kernel_size_h * 2)[:, None,
                                                            None] + conv_kernel_size_h
    msk_h_dataT = (h_offsetT >= 0) & (h_offsetT < conv_kernel_size_h)
    # h_weightsT = tl.where(msk_h_dataT, tl.exp2(-h_offsetT.to(tl.float32)), 0).to(fp16_dtype)

    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)

        m = tl.load(M + offs_m)
        m_nomsk = tl.load(M_nomsk + offs_m)
        qkT = tl.dot(k, qT)

        # Autoregressive masking, again, this is composed of three parts, the first part is the casual mask
        # the second part is the filled values, the third part is the global and local tokens,
        mask_casual = (offs_m[None, :] >= offs_n[:, None])

        msk_segment = (fill_indices[:, None] >= offs_m[None, :])

        msk_sliding_window = (offs_m[None, :] <= offs_n[:, None] + LOCAL_SEQ_MAX_LEN)

        msk_local = tl.where(local_tokens[:, None],
                             msk_segment,
                             msk_sliding_window,
                             )
        mskT = msk_local | (fill_indices == offs_n)[:, None]

        mskT = mskT & mask_casual

        pT_nomsk = tl.math.exp2(qkT - m_nomsk[None, :])
        pT = tl.math.exp2(qkT - m[None, :])

        if MASK:
            # we note that here pT_nomsk is the pT with casual masks but without additional attention masks.
            pT_nomsk = tl.where(mask_casual, pT_nomsk, 0.0)

        pT = tl.where(mskT, pT, 0.)

        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(fp16_dtype)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)

        dpt_di = dpT - Di[None, :]

        dsT = pT * dpt_di
        dsT = dsT.to(fp16_dtype)

        # now we could gather the gradients of the masks towards the end_seqs values.
        # the first is the set of the collected values

        # we compute the gradients for masks
        dmskT = pT_nomsk * dpt_di
        dmskT = dmskT.to(fp16_dtype)

        msk_seq_values = offs_m[None, :] > fill_indices[:, None]
        # both arguments are True, in this case, no grad is backpropagated towards any of the element
        msk_seq_values_tt = (fill_indices[:, None] == offs_n[:, None]) & (offs_m[None, :] <= d_endseqs_lower[:, None])

        # msk_seq_values = msk_segment | msk_seq_values_tt
        msk_seq_values = msk_seq_values & (~msk_seq_values_tt)

        d_end_seq_values0 += tl.sum(
            tl.where(
                msk_seq_values,
                dmskT, 0
            ), 1
        )

        # gradients for local tokens until the next end_seq values

        if tl.min(fill_indices) >= (start_m + BLOCK_M1 - 1):
            d_end_seq_values1 += tl.sum(
                tl.where(
                    msk_segment & mask_casual,
                    dmskT, 0
                ), 1
            )

        if curr_m < (start_n + BLOCK_N1 + LOCAL_SEQ_MAX_LEN):
            # in this case, there is no need to have gradients for local sliding windows
            d_end_seq_values2 += tl.sum(
                tl.where(
                    msk_sliding_window & mask_casual,
                    dmskT, 0
                ), 1
            )

        d_collected_values += tl.sum(
            tl.where(
                (offs_m[None, :] >= d_endseqs_lower[:, None]) & (offs_m[None, :] < d_endseqs_upper[:, None]),
                dmskT, 0
            ), 1
        )

        dmskT_masked = tl.where(mskT, dmskT, 0)

        # Since we always gather the lower left part of the attention maps, and the segment mask is located for upper
        # triangle, this results in the True values...

        if curr_m < conv_last_m:
            # TODO this is only a temporary solution, we will check if triton in the future support tensor indexing
            #  operations and update the codes accordingly!
            # for MASK is not None, although the size of BLOCK_M1 is doubled, the valid values are still involved within
            # [curr_m, conv_last_m]+ conv_kernel_size that can be fully covered by the current block. Therefore, we do
            # not need to change our codes here

            # TODO for msk == 0, we need to also have (dMSKT & fill_indices[:, None] >= offs_m[None, :])

            diag_value_ranges_involved = tl.arange(0, conv_kernel_size_h * 2)[:, None,
                                         None] + curr_m - conv_kernel_size_h

            # off_w_upper = diag_value_ranges_involved - 1
            # off_h_lower = diag_value_ranges_involved + 1

            w_offsetT = diag_value_ranges_involved - 1 - offs_n[None, :, None]
            # h_offsetT = offs_m[None, None, :] - off_h_lower

            # msk_h_dataT = (h_offsetT >= 0) & (h_offsetT < conv_kernel_size_h)
            # h_weightsT = tl.where(msk_h_dataT, tl.exp2(-h_offsetT.to(tl.float32)), 0).to(fp16_dtype)

            msk_w_dataT = (w_offsetT >= 0) & (w_offsetT < conv_kernel_size_w) & (diag_value_ranges_involved < N_CTX)
            # w_weightsT = tl.where(msk_w_dataT, tl.exp2(-w_offsetT.to(tl.float32)), 0).to(fp16_dtype)

            d_mskT_conv = tl.where(msk_w_dataT & msk_h_dataT, dmskT_masked, 0.)
            # d_mskT_conv = (dmskT * w_weightsT * h_weightsT)
            d_mskT_conv = tl.sum(
                tl.reshape(d_mskT_conv, (2 * conv_kernel_size_h, BLOCK_M1 * BLOCK_N1)), -1
            )

            diag_value_ranges_involved_ = tl.ravel(diag_value_ranges_involved)
            msk = (diag_value_ranges_involved_ >= start_n) & (
                diag_value_ranges_involved_ <= conv_last_m - conv_kernel_size_h)
            tl.store(DConvValues + conv_store_idx, d_mskT_conv, mask=msk)

            conv_store_idx += BLOCK_M1 * 4

        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok

    return dk, dv, d_end_seq_values0, d_end_seq_values1, d_end_seq_values2, d_collected_values, conv_store_idx


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
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr, LN2: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d

    end_seqs_idx_ptrs = EndSeqsIDX + offs_n * stride_end_seqs_idxn
    sliding_window_ptrs = SeqInfo + offs_n * stride_seqinfo_n + 2

    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        # Autoregressive masking.

        fill_indices = tl.load(end_seqs_idx_ptrs)
        sliding_window_tokens = tl.load(sliding_window_ptrs)

        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        casual_mask = (offs_m[:, None] >= offs_n[None, :])

        msk = (fill_indices[None, :] >= offs_m[:, None])

        msk_sliding_window = (offs_m[:, None] <= offs_n[None, :] + LOCAL_SEQ_MAX_LEN),

        msk = tl.where(sliding_window_tokens[None, :],
                       msk_sliding_window,
                       msk
                       )

        msk = msk | (fill_indices == offs_n)[None, :]
        msk = msk & casual_mask

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

        end_seqs_idx_ptrs += step_n * stride_end_seqs_idxn
        sliding_window_ptrs += step_n * stride_seqinfo_n
    return dq


configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128] \
    for BN in [32, 64] \
    for s in ([1] if is_hip() else [3, 4, 7]) \
    for w in [4, 8] \
    ]


@triton.autotune([
    triton.Config({}, num_stages=s, num_warps=w) for s in ([1] if is_hip() else [3, 4, 5, 7]) for w in [4, 8]
],
    key=['N_CTX', 'HEAD_DIM'], )
@triton.jit
def _attn_bwd(Q, K, V, SeqInfo, EndSeqsIdx, DEndSeqsLower, DEndSeqsUpper,
              sm_scale,  #
              DO,  #
              DQ, DK, DV, DSeqInfo, DCollectedValues, DConvValues,  #
              M, M_nomsk, D,
              # shared by Q/K/V/DO.
              stride_qz, stride_qh, stride_tok, stride_qd,  #
              stride_kz, stride_kh, stride_kn, stride_kd,
              stride_seqinfo_z, stride_seqinfo_h, stride_seqinfo_n, stride_seqinfo_nopt,
              stride_esq_z, stride_esq_h, stride_esq_n,
              stride_dseqinfo_z, stride_dseqinfo_h, stride_dseqinfo_n, stride_dseqinfo_nopt,
              stride_dseqcol_z, stride_dseqcol_h, stride_dseqcol_n,
              stride_dconv_z, stride_dconv_h, stride_dconv_n,
              H, N_CTX,
              LOCAL_SEQ_MAX_LEN: tl.constexpr, n_rep: tl.constexpr, kernel_size: tl.constexpr,  #
              sparse_regularized_value: tl.constexpr,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
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
    M_nomsk += off_chz
    D += off_chz

    off_h_kv = off_h // n_rep
    adj_kv = (stride_kh * off_h_kv + stride_kz * off_z).to(tl.int64)
    K += adj_kv
    V += adj_kv

    adj_end_seq = (stride_esq_h * off_h_kv + stride_esq_z * off_z).to(tl.int64)
    EndSeqsIdx += adj_end_seq
    DEndSeqsLower += adj_end_seq
    DEndSeqsUpper += adj_end_seq

    adj_seqinfo = (stride_seqinfo_z * off_z + stride_seqinfo_h * off_h_kv).to(tl.int64)
    SeqInfo += adj_seqinfo

    adj_dseqinfo = (stride_dseqinfo_z * off_z + stride_dseqinfo_h * off_h).to(tl.int64)
    DSeqInfo += adj_dseqinfo

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

    # d_end_seq_values = tl.zeros([BLOCK_N1], dtype=tl.float32)
    d_end_seq_values0 = tl.zeros([BLOCK_N1], dtype=tl.float32)
    d_end_seq_values1 = tl.zeros([BLOCK_N1], dtype=tl.float32)
    d_end_seq_values2 = tl.zeros([BLOCK_N1], dtype=tl.float32)
    d_collected_seq_values = tl.zeros([BLOCK_N1], dtype=tl.float32)

    # for conv gradients
    # compute offsets for w
    kernel_size_w: tl.constexpr = kernel_size
    kernel_size_h: tl.constexpr = kernel_size

    offs_dconv = (stride_dconv_h * off_h + stride_dconv_z * off_z).to(tl.int64)

    DConvValues = DConvValues + offs_dconv + start_n * 4

    conv_first_m = start_n - kernel_size_h
    conv_last_m = start_n + BLOCK_N1 + kernel_size_w + kernel_size_h - 1

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kd)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_kd)

    fill_indices = tl.load(EndSeqsIdx + offs_n * stride_esq_n)
    # we only need the information on whether the end_seq_value is a pure local seq or not
    local_tokens = tl.load(SeqInfo + offs_n * stride_seqinfo_n * stride_dseqinfo_nopt + 1)

    d_endseqs_lower = tl.load(DEndSeqsLower + offs_n * stride_esq_n)
    d_endseqs_upper = tl.load(DEndSeqsUpper + offs_n * stride_esq_n)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    # TODO this is only the solution for kernel_size_h == MASK_BLOCK_M1,
    #  we need to check this further for larger kernels!
    off_n_idx = pid % 2
    conv_store_idx = tl.ravel(
        tl.arange(0, kernel_size_h)[None, :] + tl.arange(0, 2)[:, None] * (kernel_size_h * 3)) - kernel_size_h
    conv_store_idx = conv_store_idx - off_n_idx * 2 * kernel_size_h

    dk, dv, d_end_seq_values0, d_end_seq_values1, d_end_seq_values2, d_collected_seq_values, conv_store_idx = _attn_bwd_dkdv(
        dk, dv, d_end_seq_values0, d_end_seq_values1, d_end_seq_values2, d_collected_seq_values, local_tokens,
        #
        DConvValues,  #
        conv_first_m, conv_last_m, conv_store_idx,  #
        kernel_size_w, kernel_size_h,  #
        Q, k, v, fill_indices, d_endseqs_lower, d_endseqs_upper, sm_scale,  #
        DO,  #
        M, M_nomsk, D, DSeqInfo,  #
        stride_tok, stride_kd,  #
        H, N_CTX, LOCAL_SEQ_MAX_LEN,  #
        MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=True  #
    )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv,  d_end_seq_values0, d_end_seq_values1, d_end_seq_values2, d_collected_seq_values, conv_store_idx = _attn_bwd_dkdv(  #
        dk, dv, d_end_seq_values0, d_end_seq_values1, d_end_seq_values2, d_collected_seq_values, local_tokens,
        #
        DConvValues,  #
        conv_first_m, conv_last_m, conv_store_idx,  #
        kernel_size_w, kernel_size_h,  #
        Q, k, v, fill_indices, d_endseqs_lower, d_endseqs_upper, sm_scale,  #
        DO,  #
        M, M_nomsk, D, DSeqInfo,  #
        stride_tok, stride_kd,  #
        H, N_CTX, LOCAL_SEQ_MAX_LEN,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=False  #
    )

    # regularized term for sparser masks.
    # our idea is that for all the pos where end_seqs is False, we encourage the item in the local masks (msk=1)
    # to provide positive gradients
    # While for all the pos where end_seqs is True, we ask their corresponding msks to provide negative values
    # This avoids the cases where all the values becomes either 1 or 0
    # Since all the local masks ends at EndSeqsIdx,
    # The amount of positive values are: N_CTX - fill_indices
    # The amount of negative values are: fill_indices -  offs_n
    d_end_seq_values0 += tl.where(offs_n == fill_indices, N_CTX - fill_indices, 0.) / N_CTX * sparse_regularized_value
    d_end_seq_values1 += tl.where(local_tokens, fill_indices - offs_n, 0.) / N_CTX * sparse_regularized_value

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_kd
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_kd
    tl.store(dk_ptrs, dk)

    # different gradients w.r.t. different values should be stored to the corresponding sides
    DSeqInfo = DSeqInfo + offs_n * stride_dseqinfo_n * stride_dseqinfo_nopt
    tl.store(DSeqInfo, d_end_seq_values0)
    tl.store(DSeqInfo + 1, d_end_seq_values1)
    tl.store(DSeqInfo + 2, d_end_seq_values2)

    tl.store(DCollectedValues + offs_n * stride_dseqcol_n, d_collected_seq_values)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
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
                      BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
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
                      BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
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
        triton.Config({'BLOCK_M': 32, }, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 16, }, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 8, }, num_stages=4, num_warps=8),

        triton.Config({'BLOCK_M': 32, }, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16, }, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 8, }, num_stages=2, num_warps=8),

        triton.Config({'BLOCK_M': 32, }, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 16, }, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 8, }, num_stages=5, num_warps=8),
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
    mask_m_store = (off_m_store < N_HEAD * N_CTX * stride_xh)

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


class NAtSAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, seq_info, end_seqs_idx, causal, sm_scale, n_rep: int = 1,
                sparse_regularized_value=0.,
                local_seq_max_length: int = 4):
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
        M_nomsk = torch.empty_like(M)

        _attn_fwd[grid](
            q, k, v, seq_info, end_seqs_idx, sm_scale, M, M_nomsk, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            seq_info.stride(0), seq_info.stride(1), seq_info.stride(2), seq_info.stride(3),  #
            end_seqs_idx.stride(0), end_seqs_idx.stride(1), end_seqs_idx.stride(2),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            LOCAL_SEQ_MAX_LEN=local_seq_max_length,
            N_REP=n_rep,
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            **extra_kern_args)
        ctx.save_for_backward(q, k, v, seq_info, end_seqs_idx, o, M, M_nomsk)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.sparse_regularized_value = sparse_regularized_value
        ctx.local_seq_max_length = local_seq_max_length
        ctx.n_rep = n_rep
        return o

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> Any:
        q, k, v, seq_info, end_seqs_idx, o, M, M_nomsk = ctx.saved_tensors
        # Since we transposed o in the attention forward pass, the memor format of do might be corrupted...
        do = do.contiguous()
        # assert do.is_contiguous()
        # assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        assert q.stride() == o.stride() == do.stride()
        n_rep = ctx.n_rep
        BATCH, N_HEAD, N_CTX, HEAD_DIM = q.shape
        N_HEAD_KV = end_seqs_idx.shape[1]
        d_endseqs_lower = torch.empty_like(end_seqs_idx)
        d_endseqs_upper = torch.empty_like(end_seqs_idx)

        grid = lambda args: (triton.cdiv(BATCH * N_HEAD_KV, args['BLOCK_M']),)
        # NOTE:
        #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
        #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
        #  - Don't forget to pass meta-parameters as keywords arguments.

        _shift_end_seqs_idx[grid](end_seqs_idx, d_endseqs_lower, d_endseqs_upper,
                                  end_seqs_idx.stride(0), end_seqs_idx.stride(1), end_seqs_idx.stride(2),
                                  BATCH, N_HEAD, N_CTX,
                                  BLOCK_N=triton.next_power_of_2(N_CTX))
        dq = torch.empty_like(q)
        # for dk, dv, we need to have [B, H_K * n_rep, N_CTX, HEAD_DIM]
        dk = torch.empty_like(q)
        dv = torch.empty_like(q)

        d_seq_info = torch.empty(
            [BATCH, N_HEAD, N_CTX, 3],
            device=seq_info.device, dtype=seq_info.dtype
        )

        # d_end_seqs_values = torch.empty([BATCH, N_HEAD, N_CTX],
        #                                device=end_seqs_info.device, dtype=end_seqs_info.dtype)
        d_end_seqs_collected = torch.empty([BATCH, N_HEAD, N_CTX],
                                           device=d_seq_info.device, dtype=d_seq_info.dtype)

        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2

        kernel_size = BLOCK_M1 // BLK_SLICE_FACTOR

        d_end_seqs_conv = torch.zeros(
            [BATCH, N_HEAD, N_CTX * 4],
            dtype=d_seq_info.dtype,
            device=d_seq_info.device,
        )

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
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)

        sparse_regularized_value = ctx.sparse_regularized_value
        local_seq_max_length = ctx.local_seq_max_length
        _attn_bwd[grid](
            q, arg_k, v, seq_info, end_seqs_idx, d_endseqs_lower, d_endseqs_upper,
            ctx.sm_scale,
            do, dq, dk, dv, d_seq_info, d_end_seqs_collected, d_end_seqs_conv,  #
            M, M_nomsk, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            seq_info.stride(0), seq_info.stride(1), seq_info.stride(2), seq_info.stride(3),  #
            end_seqs_idx.stride(0), end_seqs_idx.stride(1), end_seqs_idx.stride(2),  #
            d_seq_info.stride(0), d_seq_info.stride(1), d_seq_info.stride(2), d_seq_info.stride(3),  #
            d_end_seqs_collected.stride(0), d_end_seqs_collected.stride(1), d_end_seqs_collected.stride(2),  #
            d_end_seqs_conv.stride(0), d_end_seqs_conv.stride(1), d_end_seqs_conv.stride(2),
            N_HEAD, N_CTX, local_seq_max_length, n_rep, kernel_size,  #
            sparse_regularized_value=sparse_regularized_value,
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
        )

        # add the conv parts for non-end-seqs values
        d_end_seqs_conv = d_end_seqs_conv.view(BATCH, N_HEAD, -1, 4, kernel_size).sum(-2).view(BATCH, N_HEAD, N_CTX)
        d_seq_info[:, :, :, 0] -= d_end_seqs_conv

        # d_end_seqs_values = d_end_seqs_values.scatter_add(-1, end_seqs_idx, d_end_seqs_collected)
        # d_endseqs_lower will act as both
        if n_rep > 1:
            dk = dk.view(BATCH, N_HEAD_KV, n_rep, N_CTX, HEAD_DIM).sum(2)
            dv = dv.view(BATCH, N_HEAD_KV, n_rep, N_CTX, HEAD_DIM).sum(2)
            d_seq_info = d_seq_info.view(BATCH, N_HEAD_KV, n_rep, N_CTX, 3).sum(2)

            d_end_seqs_collected = d_end_seqs_collected.view(BATCH, N_HEAD_KV, n_rep, N_CTX).sum(2)

        d_seq_info[:, :, :, 0] = d_seq_info[:, :, :, 0].scatter_add(-1, d_endseqs_lower, -d_end_seqs_collected)
        # TODO check if this assignment is necessary!
        # d_end_seqs_values[:, :, -1] = 0.

        return dq, dk, dv, d_seq_info, None, None, None, None, None, None


def nats_attention(q, k, v, seq_info, end_seqs_idx, causal, sm_scale, n_rep: int = 1,
                        sparse_regularized_value=0., local_seq_max_length: int = 4, ):
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
        n_rep: number of replications for KV and end_seqs
        sparse_regularized_value: a regularization term controlling the
        local_seq_max_length: int, the maximal lenght of the local sequences

    Returns:
        o: torch.Tensor of shape [bsz, nheads, N_ctx, D_HEAD], attention output

    """
    return NAtSAttention.apply(q, k, v, seq_info, end_seqs_idx, causal, sm_scale, n_rep,
                                  sparse_regularized_value, local_seq_max_length)


"""
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


from nats.components.masks.triton.construct_soft_mask import _costruct_soft_mask_backward, construct_soft_mask


@triton.jit
def _costruct_hard_mask_backward(
    EndSeqsIDX,
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

    # soft_values = 1. / tl.load(EndSeqsSoft + zh_offset + offs_n, mask=n_mask, other=0)
    # collected_values = 1. / tl.load(CollectedValues + zh_offset + offs_n, mask=n_mask, other=0)
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
                grad_msk,
                0
            ), 0
        )

        # the next is for the last index, grad for log(1-soft)
        grad_soft += tl.sum(
            tl.where(
                (tl.broadcast_to(is_last_idx[None, :], (BLOCK_M, BLOCK_N))),
                -grad_msk,
                0
            ), 0
        )

        # finally, we have, grad for log(soft)
        grad_soft += tl.sum(
            tl.where(
                fill_indices[None, :] >= offs_m[:, None] + start_idx,
                0,
                grad_msk
            ), 0
        )

    tl.store(GradSeqSoft + zh_offset + offs_n, grad_soft, mask=n_mask)
    tl.store(GradCollectedValues + zh_offset + offs_n, grad_collect, mask=n_mask)


def compute_grad_soft_mask(end_seqs_idx, end_seqs_soft, grad_msk):
    start_idx = 0

    grad_soft = torch.zeros_like(end_seqs_soft)
    grad_collected = torch.zeros_like(end_seqs_soft)
    grid = lambda args: (
        triton.cdiv(end_seqs_idx.shape[2], args['BLOCK_N']),
        end_seqs_idx.shape[0] * end_seqs_idx.shape[1],
        1
    )

    _costruct_hard_mask_backward[grid](
        end_seqs_idx,  # end_seqs_soft, end_seqs_soft_collected,
        grad_soft, grad_collected,
        grad_msk, start_idx,
        end_seqs_idx.stride(0), end_seqs_idx.stride(1), end_seqs_idx.stride(2),
        grad_msk.stride(0), grad_msk.stride(1), grad_msk.stride(2), grad_msk.stride(3),
        grad_msk.shape[0], grad_msk.shape[1], grad_msk.shape[2], grad_msk.shape[3],
        BLOCK_M=256, BLOCK_N=64,
    )
    # add the gradients of collected values to soft

    # grad_soft = grad_soft.scatter_add(-1, end_seqs_idx, grad_collected)
    return grad_soft, grad_collected


def test_flash_atn():
    # TODO move this to tests!
    torch.manual_seed(0)
    from nats.components.masks.triton import construct_soft_mask, cummax_reverse
    import math
    BATCH = 1
    H = 1
    N_CTX = 256
    D_HEAD = 64
    dtype = torch.float16 if fp_16_type == 'float16' else torch.bfloat16
    device = torch.device("cuda")
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    sm_scale = 1 / math.sqrt(D_HEAD)
    from torch.nn import Transformer
    mask = Transformer.generate_square_subsequent_mask(N_CTX, device=device, dtype=dtype)
    mask_ = torch.where(mask == 0, 1., 0.)

    gumbels = (
        -torch.empty([BATCH, H, N_CTX, 3], memory_format=torch.legacy_contiguous_format).exponential_().log()
    ).cuda()

    gumbels = (
        -torch.empty([BATCH, H, N_CTX, 3], memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    #gumbels[:,:,:,0] = -torch.inf
    #gumbels[:,:,:,1] = -torch.inf
    #gumbels[:,:,:,2] = -torch.inf

    y_soft = gumbels.softmax(-1)
    index = y_soft.max(-1, keepdim=True)[1]
    y_hard = torch.zeros_like(gumbels, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
    end_seqs_hard = y_hard - y_soft.detach() + y_soft

    end_seqs_hard = end_seqs_hard.cuda().to(dtype=dtype)

    if torch.all(end_seqs_hard[:,:,:,0] == 0):
        end_seqs_indices = torch.ones([BATCH, H, N_CTX], dtype=torch.int64, ).cuda() * (N_CTX - 1)
    else:
        end_seqs_indices = N_CTX - 1 - torch.flip(torch.cummax(torch.flip(end_seqs_hard[:, :, :, 0], (-1,)), -1)[1], (-1,))
    import copy

    q0 = copy.deepcopy(q.detach())
    k0 = copy.deepcopy(k.detach())
    v0 = copy.deepcopy(v.detach())
    q0 = torch.nn.Parameter(q0, requires_grad=True)
    k0 = torch.nn.Parameter(k0, requires_grad=True)
    v0 = torch.nn.Parameter(v0, requires_grad=True)
    end_seqs_hard0 = torch.nn.Parameter(end_seqs_hard, requires_grad=True)

    res1 = segmented_attention(q0, k0, v0, end_seqs_hard0, end_seqs_indices, True, sm_scale, 1, 1e-8, 4)

    loss = (res1 ** 2).sum()
    loss.backward()

    msk1 = construct_soft_mask(end_seqs_indices, end_seqs_hard.float()[:, :, :, 0].contiguous(),
                               torch.gather(end_seqs_hard.float()[:, :, :, 0].contiguous(), -1, end_seqs_indices), 0)
    msk1 = torch.where(msk1 == 0., 1., 0.)

    local_msks = ((torch.arange(N_CTX) + 4 + 1).unsqueeze(0) > torch.arange(N_CTX).unsqueeze(1)).float().cuda()
    msk1 = torch.where(end_seqs_hard[:, :, :, -1].unsqueeze(-2) == 1., local_msks.view(1, 1, N_CTX, N_CTX), msk1)

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

    print(f'diff res max:{(res1-res2).max()}')
    print(f'diff res min:{(res1-res2).min()}')
    print(f'diff q0 grad max:{(q0.grad-q1.grad).max()}')
    print(f'diff q1 grad min:{(q0.grad-q1.grad).max()}')
    print(f'diff k0 grad max:{(k0.grad - k1.grad).max()}')
    print(f'diff k1 grad min:{(k0.grad - k1.grad).max()}')
    print(f'diff v0 grad max:{(v0.grad-v1.grad).max()}')
    print(f'diff v1 grad min:{(v0.grad-v1.grad).max()}')

    import pdb
    pdb.set_trace()

    end_seqs_hard1 = torch.nn.Parameter(end_seqs_hard, requires_grad=True)
    # end_seqs_soft1 = torch.where(end_seqs_hard1==1.0, 1.0, 1e-8)
    end_seqs_collected = torch.gather(end_seqs_hard1, -1, end_seqs_indices).float()
    q2 = copy.deepcopy(q.detach())
    k2 = copy.deepcopy(k.detach())
    v2 = copy.deepcopy(v.detach())
    q2 = torch.nn.Parameter(q2, requires_grad=True)
    k2 = torch.nn.Parameter(k2, requires_grad=True)
    v2 = torch.nn.Parameter(v2, requires_grad=True)
    end_seqs_hard1_ = end_seqs_hard1.float()
    end_seqs_soft1 = torch.lerp(torch.full_like(end_seqs_hard1_, fill_value=1e-8), end_seqs_hard1_, end_seqs_hard1_)
    end_seqs_soft1.retain_grad()

    mask_soft = construct_soft_mask(end_seqs_indices, end_seqs_soft1, end_seqs_collected)
    mask_soft.retain_grad()
    res3, qk3, qk3_exp, p3 = attention_torch_add(q2, k2, v2, mask_soft)
    loss3 = (res3 ** 2).sum()
    loss3.backward()

    from torch.nn import Transformer
    mask = Transformer.generate_square_subsequent_mask(N_CTX, device=device, dtype=dtype)
    mask_ = torch.where(mask == 0, 1., 0.)

    import pdb
    pdb.set_trace()
    ((mask1.grad - mask_soft.grad) * mask_).min()
    # import pdb
    # pdb.set_trace()

    q4 = torch.nn.Parameter(q, requires_grad=True)
    k4 = torch.nn.Parameter(k, requires_grad=True)
    v4 = torch.nn.Parameter(v, requires_grad=True)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn import functional as F

    mask2 = torch.nn.Parameter(data=mask1.data, requires_grad=True)

    res4 = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask2.bool())
    loss4 = (res4 ** 2).sum()
    loss4.backward()

    # res4 = _attention2.apply(q, k, v, True, sm_scale)
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    test_flash_atn()
    # bench_flash_attention(BATCH, N_HEADS, N_CTX, D_HEAD, True, 'bwd', 'triton', False)
    # bench_flash_attention.run(save_path=".", print_data=True)
# """
