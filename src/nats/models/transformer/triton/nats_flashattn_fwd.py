"""
This is the forward only version of the segment attention. Here we have additional boundary check to allow for any size
computation during attention forward pass.
"""
import math
import torch

import triton
import triton.language as tl

from nats.utils import check_fp16_dtype

fp_16_type = check_fp16_dtype()
fp16_dtype = tl.float16 if fp_16_type == 'float16' else tl.bfloat16


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _compute_attention_with_mask(q, k, v, mask, qk_scale,
                                 acc, l_i, m_i,
                                 offs_m: tl.constexpr, offs_n_: tl.constexpr, STAGE: tl.constexpr, fp8_v: tl.constexpr):
    qk = tl.dot(q, k)
    qk = qk * qk_scale
    if STAGE == 2:
        mask_casual = (offs_m[:, None] >= (offs_n_[None, :]))
        mask = mask & mask_casual
        qk = tl.where(mask, qk, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
    else:
        #qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
        qk = tl.where(mask, qk, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
    p = tl.math.exp2(qk)

    # p = tl.where(mask, p_nomsk, 0)

    l_ij = tl.sum(p, 1)
    # -- update m_i and l_i
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    # -- update output accumulator --
    acc = acc * alpha[:, None]
    # update acc
    if fp8_v:
        p = p.to(tl.float8e5)
    else:
        p = p.to(fp16_dtype)

    acc = tl.dot(p, v, acc)
    # update m_i and l_i
    m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _store_msks(MSK, q_msks, stride_MSKn, stride_MSKm, msk_value, offs_n_, BLOCK_M, N_CTX, load_msk):
    store_msk = (q_msks & (offs_n_ < N_CTX)[None, :]) & load_msk
    MSK_ptrs = MSK + offs_n_[None, :] * stride_MSKn + tl.arange(0, BLOCK_M)[:, None] * stride_MSKm
    tl.store(MSK_ptrs, msk_value, mask=store_msk)


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    k_ptr, v_ptr, sw_tokens_ptr, seq_idx_ptr,  #
                    start_m, qk_scale,  #
                    stride_vk, stride_vn,
                    stride_kn, stride_kk,
                    stride_SWn, stride_SeqIdxn,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    CHUNK_NAtS: tl.constexpr,  #
                    N_NAtS_BLOCK_PER_N: tl.constexpr,  #
                    LOCAL_SEQ_MAX_LEN: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, N_CTX_SEQINFO: tl.constexpr,
                    fp8_v: tl.constexpr,
                    return_masks: tl.constexpr, MSK, q_msks,
                    stride_MSKn, stride_MSKm,
                    ):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
        # we also read the first element from seq_idx_ptr to allow the local tokens know how many steps to preserve
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offs_k = tl.arange(0, HEAD_DIM)[None, :]

    start_mM = start_m * BLOCK_M
    end_n_offset = tl.arange(0, N_NAtS_BLOCK_PER_N) * tl.minimum(CHUNK_NAtS, BLOCK_N)

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_ = start_n + offs_n
        n_load_msks = offs_n_ < N_CTX

        advance_nats = start_n // CHUNK_NAtS + tl.arange(0, N_NAtS_BLOCK_PER_N)
        nats_load_mask = advance_nats < N_CTX_SEQINFO

        # -- compute qk ----
        fill_indices = tl.load(seq_idx_ptr + advance_nats, mask=nats_load_mask, other=N_CTX)
        is_global_token = fill_indices == ((start_n // CHUNK_NAtS) + tl.arange(0, N_NAtS_BLOCK_PER_N))

        is_sw_token = tl.load(sw_tokens_ptr + advance_nats, mask=nats_load_mask, other=0).to(tl.int1)

        fill_indices_scaled = fill_indices * CHUNK_NAtS + CHUNK_NAtS
        has_valid_indices = fill_indices_scaled > start_mM

        # the last element in the sw token must be larger than the corresponidng start_mN
        sw_token_end_idx = start_n + LOCAL_SEQ_MAX_LEN + end_n_offset + CHUNK_NAtS - 1
        has_valid_sw = sw_token_end_idx >= start_mM

        do_sw_compute = tl.sum((is_sw_token & has_valid_sw).to(tl.int32))
        #do_local_compute = tl.sum(((1 - is_sw_token.to(tl.int32)) & has_valid_indices).to(tl.int32))
        do_local_compute = tl.sum(((~is_sw_token) & has_valid_indices).to(tl.int32))

        do_compute = ((tl.sum(is_global_token.to(tl.int32)) + do_sw_compute + do_local_compute)).to(tl.int1)

        if do_compute:
            #"""
            #msk_local = tl.broadcast_to(fill_indices_scaled[None, :], (BLOCK_M, N_NAtS_BLOCK_PER_N)) > offs_m[:, None]
            msk_local = fill_indices_scaled[None, :] > offs_m[:, None]
            msk_sw = offs_m[:, None] <= sw_token_end_idx[None, :]
            #msk_casual = (offs_m[:, None] >= offs_n_[None, :])
            if N_NAtS_BLOCK_PER_N == 1 or CHUNK_NAtS == 1:
                msk_locals = tl.where(is_sw_token, msk_sw, msk_local)
                #msk = tl.where(is_global_token, msk_casual, msk_casual & msk_locals)
                msk = tl.where(is_global_token, 1, msk_locals)
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

            if return_masks:
                msk_casual = (offs_m[:, None] >= offs_n_[None, :])
                msk = msk_casual & msk
                _store_msks(MSK, q_msks, stride_MSKn, stride_MSKm, msk, offs_n_, BLOCK_M, N_CTX, n_load_msks[None, :])
            #"""

            k = tl.trans(tl.load(k_ptr + offs_n_[:, None] * stride_kn + offs_k, mask=n_load_msks[:, None], other=0))
            v = tl.load(v_ptr + offs_n_[:, None] * stride_vk + offs_k, mask=n_load_msks[:, None], other=0)
            acc, l_i, m_i = _compute_attention_with_mask(q, k, v, msk, qk_scale,
                                                         acc, l_i, m_i, offs_m, offs_n_, STAGE, fp8_v)

    return acc, l_i, m_i


@triton.jit
def _attn_fwd_inner_decoding_on_past(acc, l_i, m_i, q,  #
                                     k_ptr, v_ptr, sw_tokens_ptr, seq_idx_ptr,  #
                                     start_m, qk_scale,  #
                                     stride_vk, stride_vn,
                                     stride_kn, stride_kk,
                                     stride_SWn, stride_SeqIdxn,
                                     global_size, local_size, sw_token_tails, n_ctx_data,
                                     SW_TOKENS_IS_VALID,
                                     BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                                     CHUNK_NAtS: tl.constexpr,  #
                                     N_NAtS_BLOCK_PER_N: tl.constexpr,  #
                                     LOCAL_SEQ_MAX_LEN: tl.constexpr,
                                     WITH_PAST_SW_VALUES: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                                     N_CTX: tl.constexpr, N_CTX_SEQINFO: tl.constexpr, fp8_v: tl.constexpr,
                                     N_TOKENS_IN_FIRST_CHUNK: tl.constexpr,
                                     return_masks: tl.constexpr, MSK, q_msks,
                                     stride_MSKn, stride_MSKm, ):
    # This function computes the attention vlaues from the observed values
    # range of values handled by this stage
    if WITH_PAST_SW_VALUES:
        # the decoding stage that involves the past sw tokens
        lo, hi = 0, tl.cdiv(LOCAL_SEQ_MAX_LEN, BLOCK_N)
    else:
        # starting from the past sw tokens, moves until the valid size (global size + local size)
        # We note that in this case, we will not care about the feature maps from the past information
        lo, hi = tl.cdiv(LOCAL_SEQ_MAX_LEN, BLOCK_N), tl.cdiv((global_size + local_size), BLOCK_N)
    seq_idx_first = tl.load(seq_idx_ptr)
    offs_k = tl.arange(0, HEAD_DIM)[None, :]

    # loop over k, v and update accumulator
    for i in range(lo, hi):
        start_n = i * BLOCK_N
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_ = start_n + offs_n
        n_load_msks = offs_n_ < (global_size + local_size)
        # this part only contains the past observed values
        if CHUNK_NAtS > 1:
            fill_indices = seq_idx_first[None, :] * CHUNK_NAtS + CHUNK_NAtS - 1
        else:
            fill_indices = seq_idx_first[None, :]
        has_valid_local = fill_indices > start_m
        is_global_tokens = offs_n_ < global_size
        is_valid_token = n_load_msks
        do_compute = (is_global_tokens | has_valid_local) & is_valid_token
        #do_compute = (is_global_tokens+has_valid_local) & is_valid_token

        if WITH_PAST_SW_VALUES and CHUNK_NAtS == 1:
            old_token_is_start_n = offs_n_ < LOCAL_SEQ_MAX_LEN
            sw_token_is_valid = tl.load(
                SW_TOKENS_IS_VALID + offs_n_, mask=old_token_is_start_n, other=True
            ).to(tl.int1)
            # In this case, we always assume that the seen tokens is greater than sliding_window_tokens...
            # otherwise, just set tail==LOCAL_SEQ_MAX_LEN-1 in the input...
            queue_range = (offs_n_ - 1 - sw_token_tails + LOCAL_SEQ_MAX_LEN) % LOCAL_SEQ_MAX_LEN
            do_compute = tl.where(old_token_is_start_n, (queue_range > start_m) & sw_token_is_valid, do_compute)
            # past sliding window mask
        elif WITH_PAST_SW_VALUES and CHUNK_NAtS > 1:
            old_token_is_start_n = offs_n_ < LOCAL_SEQ_MAX_LEN
            nats_advance = start_n // CHUNK_NAtS + tl.arange(0, N_NAtS_BLOCK_PER_N)
            sw_token_load_msk = nats_advance < tl.cdiv(LOCAL_SEQ_MAX_LEN, CHUNK_NAtS)
            sw_token_is_valid = tl.load(
                SW_TOKENS_IS_VALID + nats_advance, mask=sw_token_load_msk, other=True
            ).to(tl.int1)
            # In this case, we always assume that the seen tokens is greater than sliding_window_tokens...
            # otherwise, just set tail==LOCAL_SEQ_MAX_LEN-1 in the input...
            # queue_range = (offs_n_ - 1 - sw_token_tails + LOCAL_SEQ_MAX_LEN) % LOCAL_SEQ_MAX_LEN
            queue_range = nats_advance * CHUNK_NAtS + CHUNK_NAtS
            queue_range = (queue_range - 1 - sw_token_tails + LOCAL_SEQ_MAX_LEN) % LOCAL_SEQ_MAX_LEN
            sw_token_is_valid = tl.broadcast_to(sw_token_is_valid[:, None], [N_NAtS_BLOCK_PER_N, CHUNK_NAtS])
            sw_token_is_valid = tl.reshape(sw_token_is_valid, [BLOCK_N])

            queue_range = tl.broadcast_to(queue_range[:, None], [N_NAtS_BLOCK_PER_N, CHUNK_NAtS])
            queue_range = tl.reshape(queue_range, [BLOCK_N])

            do_compute = tl.where(old_token_is_start_n, (queue_range > start_m) & sw_token_is_valid, do_compute)

        else:
            queue_range = offs_n_
            sw_token_is_valid = tl.zeros([BLOCK_N], dtype=tl.int1)

        do_compute = tl.sum(do_compute.to(tl.int32))
        if do_compute:
            # we first check if the mask is on the upper part of the attention maps, then we check if the mask
            # belongs to the local attention or full attention parts (identified by fill_indices)
            # finally, the global tokens must be preserved
            mask = (fill_indices[None, :] >= (offs_m[:, None]))
            mask = mask | tl.broadcast_to(is_global_tokens[None, :], (BLOCK_M, BLOCK_N))
            if WITH_PAST_SW_VALUES:
                msk_past_sw = (queue_range[None, :] >= offs_m[:, None]) & (sw_token_is_valid[None, :])
                mask = tl.where(offs_n_[None, :] < LOCAL_SEQ_MAX_LEN, msk_past_sw, mask)
            mask = mask & n_load_msks[None, :]

            if return_masks:
                _store_msks(MSK, q_msks, stride_MSKn, stride_MSKm, mask, offs_n_, BLOCK_M, N_CTX, n_load_msks[None, :])

            k = tl.trans(tl.load(k_ptr + offs_n_[:, None] * stride_kn + offs_k, mask=n_load_msks[:, None], other=0))
            v = tl.load(v_ptr + offs_n_[:, None] * stride_vk + offs_k, mask=n_load_msks[:, None], other=0)
            acc, l_i, m_i = _compute_attention_with_mask(q, k, v, mask, qk_scale,
                                                         acc, l_i, m_i, offs_m, offs_n_, 0, fp8_v)

    return acc, l_i, m_i


@triton.jit
def _attn_fwd_inner_decoding_on_current(acc, l_i, m_i, q,  #
                                        k_ptr, v_ptr, sw_tokens_ptr, seq_idx_ptr,  #
                                        start_m, qk_scale,  #
                                        stride_vk, stride_vn,
                                        stride_kn, stride_kk,
                                        stride_SWn, stride_SeqIdxn,
                                        global_size, local_size, sw_token_tails, n_ctx_data,
                                        SW_TOKENS_IS_VALID,
                                        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                                        CHUNK_NAtS: tl.constexpr,  #
                                        N_NAtS_BLOCK_PER_N: tl.constexpr,  #
                                        LOCAL_SEQ_MAX_LEN: tl.constexpr,
                                        STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                                        N_CTX: tl.constexpr, N_CTX_SEQINFO: tl.constexpr, fp8_v: tl.constexpr,
                                        N_TOKENS_IN_FIRST_CHUNK: tl.constexpr,
                                        return_masks: tl.constexpr, MSK, q_msks,
                                        stride_MSKn, stride_MSKm,
                                        ):
    start_m_ = start_m - N_TOKENS_IN_FIRST_CHUNK
    if STAGE == 1:
        lo, hi = n_ctx_data // BLOCK_N, (n_ctx_data + start_m_ * BLOCK_M) // BLOCK_N
        # we also read the first element from seq_idx_ptr to allow the local tokens know how many steps to preserve
    elif STAGE == 2:
        lo, hi = (n_ctx_data + start_m_ * BLOCK_M) // BLOCK_N, tl.cdiv(n_ctx_data + (start_m + 1) * BLOCK_M, BLOCK_N)
    # causal = False
    else:
        lo, hi = n_ctx_data // BLOCK_N, tl.cdiv(N_CTX, BLOCK_N)

    offs_k = tl.arange(0, HEAD_DIM)[None, :]
    n_ctx_data = tl.multiple_of(n_ctx_data, CHUNK_NAtS)
    # loop over k, v and update accumulator

    end_n_offset = tl.arange(0, N_NAtS_BLOCK_PER_N) * tl.minimum(CHUNK_NAtS, BLOCK_N)

    for i in range(lo, hi):
        start_n = i * BLOCK_N
        start_n = tl.multiple_of(start_n, BLOCK_N)

        curr_n = start_n - n_ctx_data

        offs_n_ = start_n + offs_n
        # in this case, we need to load the new token states
        advance_nats = curr_n // CHUNK_NAtS + tl.arange(0, N_NAtS_BLOCK_PER_N)
        # Since n_ctx_data is always the multiple of CHUNK_NAtS, if N_NAtS_BLOCK_PER_N == 1, then CHUNK_NAtS >= BLOCK_N,
        # In this case, curr_n should be >= 0. Otherwise, we could always guarantee that at least one element is valid
        load_msk_seq_info = (advance_nats >= 0) & (advance_nats < N_CTX_SEQINFO)

        fill_indices = tl.load(seq_idx_ptr + advance_nats, mask=load_msk_seq_info, other=N_CTX_SEQINFO)
        fill_indices_scaled = fill_indices * CHUNK_NAtS + CHUNK_NAtS

        is_sw_token = tl.load(sw_tokens_ptr + advance_nats, mask=load_msk_seq_info, other=0).to(tl.int1)
        is_global_token = ((fill_indices == advance_nats) & (advance_nats >= 0))

        has_valid_indices = fill_indices_scaled > start_m

        sw_token_end_idx = curr_n + LOCAL_SEQ_MAX_LEN + end_n_offset + CHUNK_NAtS - 1
        has_valid_sw = sw_token_end_idx >= start_m

        do_sw_compute = tl.sum((is_sw_token & has_valid_sw).to(tl.int32))
        do_local_compute = tl.sum(((~is_sw_token) & has_valid_indices).to(tl.int32))
        #do_local_compute = tl.sum(((1-is_sw_token.to(tl.int32)).to(tl.int1) & has_valid_indices).to(tl.int32))
        do_compute = (tl.sum(is_global_token.to(tl.int32)) + do_sw_compute + do_local_compute).to(tl.int1)

        """

        load_msk = (load_offset >= 0) & (load_offset < N_CTX)
        fill_indices = tl.load(seq_idx_ptr + load_offset, mask=load_msk, other=N_CTX)
        sliding_window_tokens = tl.load(sw_tokens_ptr + load_offset, mask=load_msk, other=0).to(tl.int1)

        has_valid_local = fill_indices > start_m
        has_valid_sw = start_n + LOCAL_SEQ_MAX_LEN + BLOCK_N > start_m
        is_global_tokens = ((fill_indices == load_offset) & (load_offset >= 0))
        is_valid_token = load_offset >= 0
        # we only have four types of tokens: global tokens, invalid tokens, sliding window tokens and local
        # tokens
        do_compute = is_global_tokens | tl.where(sliding_window_tokens, has_valid_sw, has_valid_local)
        do_compute = do_compute & is_valid_token
        #"""

        if do_compute:
            load_offset = offs_n_ - n_ctx_data
            load_msk = (load_offset >= 0) & (offs_n_ < N_CTX)
            # we first check if the mask is on the upper part of the attention maps, then we check if the mask
            # belongs to the local attention or full attention parts (identified by fill_indices)
            # finally, the global tokens must be preserved
            msk_local = tl.broadcast_to(fill_indices_scaled[None, :], (BLOCK_M, N_NAtS_BLOCK_PER_N)) > offs_m[:, None]
            msk_sw = offs_m[:, None] <= sw_token_end_idx[None, :]

            if N_NAtS_BLOCK_PER_N == 1 or CHUNK_NAtS == 1:
                msk_locals = tl.where(is_sw_token, msk_sw, msk_local)
                mask = tl.where(is_global_token, 1, msk_locals)
            else:
                #
                msk_local = tl.reshape(msk_local, [BLOCK_M, N_NAtS_BLOCK_PER_N, 1])
                msk_sw = tl.reshape(msk_sw, [BLOCK_M, N_NAtS_BLOCK_PER_N, 1])

                msk_locals = tl.where(is_sw_token[None, :, None], msk_sw, msk_local)
                mask = tl.where(is_global_token[None, :, None], 1, msk_locals)
                mask = tl.broadcast_to(mask, [BLOCK_M, N_NAtS_BLOCK_PER_N, CHUNK_NAtS])
                mask = tl.reshape(mask, [BLOCK_M, BLOCK_N])

            mask = mask & load_msk[None, :]
            if return_masks:
                _store_msks(MSK, q_msks, stride_MSKn, stride_MSKm, mask, offs_n_, BLOCK_M, N_CTX, load_msk[None, :])

            k = tl.trans(tl.load(k_ptr + offs_n_[:, None] * stride_kn + offs_k, mask=load_msk[:, None], other=0))
            v = tl.load(v_ptr + offs_n_[:, None] * stride_vk + offs_k, mask=load_msk[:, None], other=0)

            acc, l_i, m_i = _compute_attention_with_mask(q, k, v, mask, qk_scale,
                                                         acc, l_i, m_i, offs_m, load_offset, STAGE, fp8_v)

    return acc, l_i, m_i


configs = [
    triton.Config({
        'BLOCK_M': BM,
        # 'BLOCK_N': BN
    }, num_stages=s, num_warps=w) \
    for BM in [64, 128, 256] \
    # for BN in [32, 64] \
    # for s in ([1] if is_hip() else [3, 4, 7]) \
    # for w in [4, 8] \
    #for BM in [ 64, ] \
    # for BN in [32, ] \
    for s in ([1] if is_hip() else [3, 4, 7]) \
    for w in [4, 8] \
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = 32
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, sliding_tokens, seq_idx, sm_scale, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_SWz, stride_SWh, stride_SWn,  #
              stride_SeqIdxz, stride_SeqIdxh, stride_SeqIdxn,  #
              # the following variables are optional values containing important past information on kv caches
              GlobalSize, LocalSize, SW_TOKENS_IS_VALID, sw_token_tails, n_ctx_data,
              stride_GSz, stride_GSh,
              stride_SWTVz, stride_SWTVh, stride_SWTVn,
              Z, H, N_CTX, N_CTX_Q, N_CTX_SEQINFO,  #
              HAS_PAST_INFO: tl.constexpr,
              LOCAL_SEQ_MAX_LEN: tl.constexpr,
              N_REP: tl.constexpr,
              HEAD_DIM: tl.constexpr,  #
              return_masks: tl.constexpr,
              MSK,
              stride_MSKz, stride_MSKh, stride_MSKm, stride_MSKn,
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              CHUNK_NAtS: tl.constexpr,  #
              N_NAtS_BLOCK_PER_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              N_TOKENS_IN_FIRST_CHUNK: tl.constexpr,
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    off_h_kv = off_h // N_REP
    kv_offset = off_z.to(tl.int64) * stride_kz + off_h_kv.to(tl.int64) * stride_kh
    seq_idx_offset = off_z.to(tl.int64) * stride_SeqIdxz + off_h_kv.to(tl.int64) * stride_SeqIdxh
    sw_offset = off_z.to(tl.int64) * stride_SWz + off_h_kv.to(tl.int64) * stride_SWh

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh + start_m * BLOCK_M * stride_qm
    q_offsets = tl.arange(0, HEAD_DIM)[None, :] * stride_qk + tl.arange(0, BLOCK_M)[:, None] * stride_qm
    q_msks = ((start_m * BLOCK_M + tl.arange(0, BLOCK_M)) < N_CTX_Q)[:, None]

    Out = Out + q_offset
    q = tl.load(Q + q_offset + q_offsets, mask=q_msks, other=0)

    v_ptr = V + kv_offset
    k_ptr = K + kv_offset
    seq_idx_ptr = seq_idx + seq_idx_offset
    sw_tokens_ptr = sliding_tokens + sw_offset

    if return_masks:
        msk_offset = off_z.to(tl.int64) * stride_MSKz + off_h_kv.to(tl.int64) * stride_MSKh
        MSK = MSK + msk_offset + start_m.to(tl.int64) * BLOCK_M * stride_MSKm

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if HAS_PAST_INFO:
        GS_offset = off_z.to(tl.int64) * stride_GSz + off_h_kv.to(tl.int64) * stride_GSh
        SW_TOKENS_IS_VALID_offset = off_z.to(tl.int64) * stride_SWTVz + off_h_kv.to(tl.int64) * stride_SWTVh

        global_size = tl.load(GlobalSize + GS_offset).to(tl.int64)  # global value for the current head
        local_size = tl.load(LocalSize + GS_offset).to(tl.int64)  # local value for the current head
        n_ctx_data = n_ctx_data.to(tl.int64)

        SW_TOKENS_IS_VALID = SW_TOKENS_IS_VALID + SW_TOKENS_IS_VALID_offset

        # we would always assume that the decoding with past information is the causal encoding...
        # the first stage, we have past sw tokens

        start_m = start_m + N_TOKENS_IN_FIRST_CHUNK
        offs_m = offs_m + N_TOKENS_IN_FIRST_CHUNK
        #"""
        acc, l_i, m_i = _attn_fwd_inner_decoding_on_past(acc, l_i, m_i, q, k_ptr, v_ptr,
                                                         sw_tokens_ptr, seq_idx_ptr,  #
                                                         start_m, qk_scale,  #
                                                         stride_vk, stride_vn,
                                                         stride_kn, stride_kk,
                                                         stride_SWn, stride_SeqIdxn,
                                                         global_size, local_size, sw_token_tails, n_ctx_data,
                                                         SW_TOKENS_IS_VALID,
                                                         BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                                         CHUNK_NAtS,  #
                                                         N_NAtS_BLOCK_PER_N,  #
                                                         LOCAL_SEQ_MAX_LEN,
                                                         True, offs_m, offs_n, N_CTX, N_CTX_SEQINFO,
                                                         V.dtype.element_ty == tl.float8e5,
                                                         N_TOKENS_IN_FIRST_CHUNK=N_TOKENS_IN_FIRST_CHUNK,
                                                         return_masks=return_masks, MSK=MSK, q_msks=q_msks,
                                                         stride_MSKn=stride_MSKn, stride_MSKm=stride_MSKm,
                                                         )
        # the second stage, we move until global_size + local_size
        acc, l_i, m_i = _attn_fwd_inner_decoding_on_past(acc, l_i, m_i, q, k_ptr, v_ptr,
                                                         sw_tokens_ptr, seq_idx_ptr,  #
                                                         start_m, qk_scale,  #
                                                         stride_vk, stride_vn,
                                                         stride_kn, stride_kk,
                                                         stride_SWn, stride_SeqIdxn,
                                                         global_size, local_size, sw_token_tails, n_ctx_data,
                                                         SW_TOKENS_IS_VALID,
                                                         BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                                         CHUNK_NAtS,  #
                                                         N_NAtS_BLOCK_PER_N,  #
                                                         LOCAL_SEQ_MAX_LEN,
                                                         False, offs_m, offs_n, N_CTX, N_CTX_SEQINFO,
                                                         V.dtype.element_ty == tl.float8e5,
                                                         N_TOKENS_IN_FIRST_CHUNK=N_TOKENS_IN_FIRST_CHUNK,
                                                         return_masks=return_masks, MSK=MSK, q_msks=q_msks,
                                                         stride_MSKn=stride_MSKn, stride_MSKm=stride_MSKm,
                                                         )
        # we now work with the current tokens
        # """
        # the thrid stage, we start from n_ctx

        if STAGE & 1:
            acc, l_i, m_i = _attn_fwd_inner_decoding_on_current(acc, l_i, m_i, q, k_ptr, v_ptr,
                                                                sw_tokens_ptr, seq_idx_ptr,  #
                                                                start_m, qk_scale,  #
                                                                stride_vk, stride_vn,
                                                                stride_kn, stride_kk,
                                                                stride_SWn, stride_SeqIdxn,
                                                                global_size, local_size, sw_token_tails, n_ctx_data,
                                                                SW_TOKENS_IS_VALID,
                                                                BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                                                CHUNK_NAtS,  #
                                                                N_NAtS_BLOCK_PER_N,  #
                                                                LOCAL_SEQ_MAX_LEN,
                                                                4 - STAGE, offs_m, offs_n, N_CTX, N_CTX_SEQINFO,
                                                                V.dtype.element_ty == tl.float8e5,
                                                                N_TOKENS_IN_FIRST_CHUNK=N_TOKENS_IN_FIRST_CHUNK,
                                                                return_masks=return_masks, MSK=MSK, q_msks=q_msks,
                                                                stride_MSKn=stride_MSKn, stride_MSKm=stride_MSKm,
                                                                )

        if STAGE & 2:
            acc, l_i, m_i = _attn_fwd_inner_decoding_on_current(acc, l_i, m_i, q, k_ptr, v_ptr,
                                                                sw_tokens_ptr, seq_idx_ptr,  #
                                                                start_m, qk_scale,  #
                                                                stride_vk, stride_vn,
                                                                stride_kn, stride_kk,
                                                                stride_SWn, stride_SeqIdxn,
                                                                global_size, local_size, sw_token_tails, n_ctx_data,
                                                                SW_TOKENS_IS_VALID,
                                                                BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                                                CHUNK_NAtS,  #
                                                                N_NAtS_BLOCK_PER_N,  #
                                                                LOCAL_SEQ_MAX_LEN,
                                                                2, offs_m, offs_n, N_CTX, N_CTX_SEQINFO,
                                                                V.dtype.element_ty == tl.float8e5,
                                                                N_TOKENS_IN_FIRST_CHUNK=N_TOKENS_IN_FIRST_CHUNK,
                                                                return_masks=return_masks, MSK=MSK, q_msks=q_msks,
                                                                stride_MSKn=stride_MSKn, stride_MSKm=stride_MSKm,
                                                                )
        # """
    else:
        if STAGE & 1:
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptr, v_ptr,
                                            sw_tokens_ptr, seq_idx_ptr,  #
                                            start_m, qk_scale,  #
                                            stride_vk, stride_vn,
                                            stride_kn, stride_kk,
                                            stride_SWn, stride_SeqIdxn,
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            CHUNK_NAtS,  #
                                            N_NAtS_BLOCK_PER_N,  #
                                            LOCAL_SEQ_MAX_LEN,
                                            4 - STAGE, offs_m, offs_n, N_CTX, N_CTX_SEQINFO,
                                            fp8_v=V.dtype.element_ty == tl.float8e5,
                                            return_masks=return_masks, MSK=MSK, q_msks=q_msks,
                                            stride_MSKn=stride_MSKn, stride_MSKm=stride_MSKm,
                                            )

        # stage 2: on-band
        if STAGE & 2:
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptr, v_ptr,
                                            sw_tokens_ptr, seq_idx_ptr,  #
                                            start_m, qk_scale,  #
                                            stride_vk, stride_vn,
                                            stride_kn, stride_kk,
                                            stride_SWn, stride_SeqIdxn,
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            CHUNK_NAtS,  #
                                            N_NAtS_BLOCK_PER_N,  #
                                            LOCAL_SEQ_MAX_LEN,
                                            2, offs_m, offs_n, N_CTX, N_CTX_SEQINFO,
                                            fp8_v=V.dtype.element_ty == tl.float8e5,  #
                                            return_masks=return_masks, MSK=MSK, q_msks=q_msks,
                                            stride_MSKn=stride_MSKn, stride_MSKm=stride_MSKm,
                                            )
    # epilogue
    acc = acc / l_i[:, None]
    o_ptr = Out + tl.arange(0, HEAD_DIM)[None, :] * stride_on + tl.arange(0, BLOCK_M)[:, None] * stride_om

    tl.store(o_ptr, acc.to(Out.type.element_ty), mask=q_msks, )


def nats_prefill(q, k, v, end_seqs_idx, sliding_tokens, sm_scale,
                 nats_chunk_size: int = 1,
                 global_kv_size: torch.Tensor | None = None,
                 local_kv_size: torch.Tensor | None = None,
                 valid_sw_tokens: torch.Tensor | None = None,
                 n_ctx_data: torch.Tensor | None = None,
                 sw_token_tails: torch.Tensor | None = None,
                 n_rep: int = 1,
                 n_tokens_in_first_chunk: int=0,
                 local_seq_max_length: int = 4,
                 return_masks: bool = False,
                 block_n=32
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
    N_CTX = k.shape[2]
    N_CTX_Q = q.shape[2]
    if return_masks:
        mask = torch.zeros([k.shape[0], k.shape[1], q.shape[2], k.shape[2]], dtype=torch.bool, device=q.device)
        msk_kwargs = {
            "MSK": mask,
            "stride_MSKz": mask.stride(0),
            "stride_MSKh": mask.stride(1),
            "stride_MSKm": mask.stride(2),
            "stride_MSKn": mask.stride(3),
        }
    else:
        mask = None
        msk_kwargs = {
            "MSK": None,
            "stride_MSKz": None,
            "stride_MSKh": None,
            "stride_MSKm": None,
            "stride_MSKn": None,
        }

    if global_kv_size is None:
        # vanilla forward attention without considering the
        assert N_CTX == N_CTX_Q
        _attn_fwd[grid](
            q, k, v, sliding_tokens, end_seqs_idx, sm_scale, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            sliding_tokens.stride(0), sliding_tokens.stride(1), sliding_tokens.stride(2),  #
            end_seqs_idx.stride(0), end_seqs_idx.stride(1), end_seqs_idx.stride(2),  #
            global_kv_size, local_kv_size, valid_sw_tokens, sw_token_tails, n_ctx_data,
            None, None, None, None, None,
            HAS_PAST_INFO=False,
            Z=q.shape[0], H=q.shape[1],  #
            N_CTX=N_CTX, N_CTX_Q=N_CTX_Q, N_CTX_SEQINFO=end_seqs_idx.shape[-1],  #
            LOCAL_SEQ_MAX_LEN=local_seq_max_length,
            N_TOKENS_IN_FIRST_CHUNK=0,
            N_REP=n_rep,
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            return_masks=return_masks,
            BLOCK_N=block_n,
            CHUNK_NAtS=nats_chunk_size,
            N_NAtS_BLOCK_PER_N=math.ceil(block_n / nats_chunk_size),
            **msk_kwargs,
            **extra_kern_args)
    else:
        if nats_chunk_size > 1:
            # in the chunk case, we set the
            n_ctx_data = n_ctx_data // nats_chunk_size * nats_chunk_size
        _attn_fwd[grid](
            q, k, v, sliding_tokens, end_seqs_idx, sm_scale, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            sliding_tokens.stride(0), sliding_tokens.stride(1), sliding_tokens.stride(2),  #
            end_seqs_idx.stride(0), end_seqs_idx.stride(1), end_seqs_idx.stride(2),  #
            global_kv_size, local_kv_size, valid_sw_tokens, sw_token_tails,
            n_ctx_data.item() if isinstance(n_ctx_data, torch.Tensor) else n_ctx_data,
            global_kv_size.stride(0), global_kv_size.stride(1),
            valid_sw_tokens.stride(0), valid_sw_tokens.stride(1), valid_sw_tokens.stride(2),
            HAS_PAST_INFO=True,
            Z=q.shape[0], H=q.shape[1],  #
            N_CTX=N_CTX, N_CTX_Q=N_CTX_Q, N_CTX_SEQINFO=end_seqs_idx.shape[-1],  #
            LOCAL_SEQ_MAX_LEN=local_seq_max_length,
            N_TOKENS_IN_FIRST_CHUNK=n_tokens_in_first_chunk,
            N_REP=n_rep,
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            return_masks=return_masks,
            BLOCK_N=block_n,
            N_NAtS_BLOCK_PER_N=math.ceil(block_n / nats_chunk_size),
            CHUNK_NAtS=nats_chunk_size,
            **msk_kwargs,
            **extra_kern_args)
    if return_masks:
        mask = mask.tril_(N_CTX - N_CTX_Q)
    return o, mask


# """
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


def test_flash_attn():
    # TODO move this to tests!
    torch.manual_seed(0)
    from nats.components.masks.triton import construct_soft_mask, cummax_reverse
    import math
    BATCH = 4
    H = 8
    N_CTX = 507
    D_HEAD = 64
    dtype = torch.float16 if fp_16_type == 'float16' else torch.bfloat16
    device = torch.device("cuda")
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)

    # q = torch.ones_like(q) * torch.arange((N_CTX)).view(1,1,-1,1).cuda()
    # k =  torch.ones_like(k) * torch.arange((N_CTX)).view(1,1,-1,1).cuda()
    # v =  torch.ones_like(v) * torch.arange((N_CTX)).view(1,1,-1,1).cuda()

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
    # gumbels[:,:,:,0] = -torch.inf
    # gumbels[:,:,:,1] = -torch.inf
    # gumbels[:,:,:,2] = -torch.inf

    y_soft = gumbels.softmax(-1)
    index = y_soft.max(-1, keepdim=True)[1]
    index[:, :, -1] = 0
    y_hard = torch.zeros_like(gumbels, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
    end_seqs_hard = y_hard - y_soft.detach() + y_soft
    end_seqs_hard = end_seqs_hard.cuda().to(dtype=dtype)

    if torch.all(end_seqs_hard[:, :, :, 0] == 0):
        end_seqs_indices = torch.ones([BATCH, H, N_CTX], dtype=torch.int64, ).cuda() * (N_CTX - 1)
    else:
        end_seqs_indices = N_CTX - 1 - torch.flip(torch.cummax(torch.flip(end_seqs_hard[:, :, :, 0], (-1,)), -1)[1],
                                                  (-1,))
    import copy

    q0 = copy.deepcopy(q.detach())
    k0 = copy.deepcopy(k.detach())
    v0 = copy.deepcopy(v.detach())
    q0 = torch.nn.Parameter(q0, requires_grad=True)
    k0 = torch.nn.Parameter(k0, requires_grad=True)
    v0 = torch.nn.Parameter(v0, requires_grad=True)
    end_seqs_hard0 = torch.nn.Parameter(end_seqs_hard, requires_grad=True)

    res1, msk = nats_prefill(q0.contiguous(), k0.contiguous(), v0.contiguous(), end_seqs_indices,
                             y_hard[..., 2].cuda().contiguous(), sm_scale, n_rep=1, local_seq_max_length=4,
                             return_masks=True)

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

    # end_seq_soft_grad, end_seq_collected_grad = compute_grad_soft_mask(end_seqs_indices, end_seqs_hard, mask1.grad)

    print(f'diff res max:{(res1 - res2).max()}')
    print(f'diff res min:{(res1 - res2).min()}')


def test_flatsh_attn_after_update():
    torch.manual_seed(101)

    from nats.components.cache.nats_chunk_layer_cache import NAtSChunkLayerCache
    from nats.components.cache.nats_layer_cache import NAtSLayerCache
    import math
    BATCH = 4
    H = 8
    N_CTX = 507
    D_HEAD = 64
    SW_SIZE = 32
    store_first = 507
    nats_chunk_size = 1
    dtype = torch.float16 if fp_16_type == 'float16' else torch.bfloat16
    device = torch.device("cuda")
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)

    nseqinfo1 = math.ceil(N_CTX / nats_chunk_size)
    gumbels = (
        -torch.empty([BATCH, H, nseqinfo1, 3], memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    y_soft = gumbels.softmax(-1)
    index = y_soft.max(-1, keepdim=True)[1]
    index[:, :, -1] = 0

    y_hard = torch.zeros_like(gumbels, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0).cuda()
    # now we want to ensure that after one cache update, the results are the same
    if nats_chunk_size == 1:
        cache = NAtSLayerCache(bsz=BATCH, n_msks=H, sliding_window_size=SW_SIZE, )
    else:
        cache = NAtSChunkLayerCache(bsz=BATCH, n_msks=H, sliding_window_size=SW_SIZE, chunk_size=nats_chunk_size)
    end_seq_idx = nseqinfo1 - 1 - torch.flip(torch.cummax(torch.flip(y_hard[..., 0], (-1,)), -1)[1], (-1,))

    cache.update(key_states=k[:, :, :store_first].contiguous(), value_states=v[:, :, :store_first].contiguous(),
                 token_states_info=y_hard[:, :, :store_first].contiguous())
    """
    res_start1, msk_start = nats_prefill(q, k, v,
                                         sliding_tokens=y_hard[..., 2].contiguous(),
                                         nats_chunk_size=nats_chunk_size,
                                         end_seqs_idx=end_seq_idx,
                                         sm_scale=1 / math.sqrt(D_HEAD),
                                         n_rep=1,
                                         return_masks=True,
                                         local_seq_max_length=SW_SIZE,
                                         )
    mask_start = cache.generate_mask(N_CTX)

    mask_start = mask_start.to(q)
    res2_start, qk, qk2_exp, p = attention_torch(q, k, v, mask=mask_start)
    if msk_start is not None:
        msk_start = msk_start.to(mask_start)
        msk_diff1 = msk_start - mask_start
        print(f"msk diff past: {msk_diff1.abs().sum()}")

    print(f'diff res max:{(res_start1 - res2_start).max()}')
    print(f'diff res min:{(res_start1 - res2_start).min()}')
    import pdb
    pdb.set_trace()
    #"""
    cache.post_update(store_first)

    if store_first < N_CTX:
        cache.update(key_states=k[:, :, store_first:].contiguous(), value_states=v[:, :, store_first:].contiguous(),
                     token_states_info=y_hard[:, :, store_first:].contiguous())
        cache.post_update(N_CTX - store_first)

    N_CTX2 = 54
    q2 = torch.randn((BATCH, H, N_CTX2, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    k2 = torch.randn((BATCH, H, N_CTX2, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    v2 = torch.randn((BATCH, H, N_CTX2, D_HEAD), dtype=dtype, device=device, requires_grad=True)

    n_remains = N_CTX - N_CTX // nats_chunk_size * nats_chunk_size
    n_endseq2 = math.ceil((N_CTX2 + n_remains) / nats_chunk_size)

    gumbels2 = (
        -torch.empty([BATCH, H, n_endseq2, 3], memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    y_soft2 = gumbels2.softmax(-1)
    index2 = y_soft2.max(-1, keepdim=True)[1]
    index2[:, :, -1] = 0
    y_hard2 = torch.zeros_like(gumbels2, memory_format=torch.legacy_contiguous_format).scatter_(-1, index2, 1.0).cuda()

    k2, v2 = cache.update(key_states=k2, value_states=v2, token_states_info=y_hard2)
    global_size = cache.size_global_kv
    local_size = cache.size_local_kv
    is_valid_sw_tokens = cache.valid_tokens[..., :SW_SIZE:nats_chunk_size].contiguous()
    n_ctx_data_chunk = cache.n_ctx_data // nats_chunk_size * nats_chunk_size

    mask = cache.generate_mask(N_CTX2)

    end_seqs_info = y_hard2[..., 0].clone().contiguous()
    end_seqs_info[..., -1] = 1.
    end_seq_idx = n_endseq2 - 1 - torch.flip(torch.cummax(torch.flip(end_seqs_info, (-1,)), -1)[1], (-1,))

    res1, msk = nats_prefill(q2, k2, v2,
                             sliding_tokens=y_hard2[..., 2].contiguous(),
                             nats_chunk_size=nats_chunk_size,
                             end_seqs_idx=end_seq_idx,
                             sm_scale=1 / math.sqrt(D_HEAD),
                             n_rep=1,
                             n_tokens_in_first_chunk=cache.chunk_fill_size - N_CTX2 if nats_chunk_size > 1 else 0,
                             return_masks=True,
                             local_seq_max_length=SW_SIZE,
                             global_kv_size=global_size,
                             local_kv_size=local_size,
                             valid_sw_tokens=is_valid_sw_tokens,
                             n_ctx_data=cache.n_ctx_data,
                             sw_token_tails=cache.sliding_queue_tail
                             )
    mask = mask.to(q2)
    res2, qk, qk2_exp, p = attention_torch(q2, k2, v2, mask=mask)
    if msk is not None:
        msk = msk.to(mask)
        n_ctx_data_chunk = cache.n_ctx_data // nats_chunk_size * nats_chunk_size
        msk_diff1 = msk[:, :, :, n_ctx_data_chunk:] - mask[:, :, :, n_ctx_data_chunk:]
        msk_diff2 = msk[:, :, :, SW_SIZE:n_ctx_data_chunk] - mask[:, :, :, SW_SIZE:n_ctx_data_chunk]
        msk_diffsw = msk[...,:SW_SIZE] - mask[...,:SW_SIZE]

        print(f"msk diff future: {msk_diff1.abs().sum()}")
        print(f"msk diff past: {msk_diff2.abs().sum()}")
        print(f"msk diff sw: {msk_diffsw.abs().sum()}")

    print(f'diff res max:{(res1 - res2).max()}')
    print(f'diff res min:{(res1 - res2).min()}')

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    # test_flash_attn()
    test_flatsh_attn_after_update()
    # bench_flash_attention(BATCH, N_HEADS, N_CTX, D_HEAD, True, 'bwd', 'triton', False)
    # bench_flash_attention.run(save_path=".", print_data=True)
# """
