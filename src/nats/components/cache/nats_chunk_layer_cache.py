import math
import pdb

import torch
import torch.nn.functional as F

from nats.chunk_utils import get_pad_size
from nats.components.cache.cache_update import (
    CacheUpdateInfo,
    merge_local_to_global_caches
)
from nats.components.masks.utils import generate_hard_seq_masks


class NAtSChunkLayerCache:
    def __init__(self, bsz: int, n_msks: int, sliding_window_size: int,
                 layer_id: int = 0,
                 chunk_size: int = 16,
                 device: torch.device = torch.device('cuda')):
        """
        Cache for each layer from segment transformer. Currently, we have three types of tokens: the global tokens, the
        local tokens, and sliding window tokens (with fixed length). The size of sliding window tokens with is always
        fixed, therefore, we could consider them as part of global tokens and always put them in the beginning of the
        global tokens. We note that the head of global tokens only store the `past` sliding window tokens, so their
        size should be sliding_window - 1
        """
        # the sliding window only store the `past` sliding
        # sliding_window_size = sliding_window_size - 1
        self.layer_id = layer_id

        self.size_local_kv = torch.zeros(bsz, n_msks, device=device, dtype=torch.long)
        self.size_global_kv = torch.full([bsz, n_msks], fill_value=sliding_window_size, device=device, dtype=torch.long)

        # used for identify the block size
        self.chunk_size = chunk_size
        self.chunk_fill_size = 0
        self.chunk_fill_size_last = 0
        # this value is used to store the intermediate mean values from the input features
        self.cumulative_x_value = None

        # since we restrict the maximal local size, we store the local cache as a queue, this value indicates the
        # tail of the local cache.
        self.sliding_window_size = sliding_window_size
        # in principle, all the values in the same batch should share the same tails
        self.sliding_queue_tail = 0
        # TODO check if this is necesasry...
        assert self.sliding_window_size % self.chunk_size == 0
        self.n_chunks_per_sw = self.sliding_window_size // self.chunk_size

        # this value is only used for  recoding the most recent sliding window info
        self._tokens_info = None
        self._cache_k_new = None
        self._cache_v_new = None
        self._n_global_min = torch.min(self.size_global_kv)
        self._sliding_window_tokens = None

        self.new_token_states = None

        self.n_ctx_local = torch.max(self.size_local_kv)
        # self.n_ctx_global = torch.max(self.size_global_kv)

        self.local_tokens = None

        self.n_ctx_data = torch.max(self.size_global_kv + self.size_local_kv)

        self._end_seqs_local = None
        self.valid_tokens = None

        self.cache_k = None
        self.cache_v = None

        self.bsz = bsz
        self.n_msks = n_msks

        self._seen_tokens = 0

        self.device = device

    def split(self, split_size: list[int]) -> tuple['NAtSChunkLayerCache']:
        """
        Split the cache into two seperated ones to reduce some computation overheads
        """
        new_caches = tuple(
            NAtSChunkLayerCache(
                bsz=self.bsz,
                n_msks=n_msk,
                sliding_window_size=self.sliding_window_size,
                layer_id=self.layer_id, device=self.device
            ) for n_msk in split_size)

        size_local_kv_split = torch.split(self.size_local_kv, split_size, dim=1)
        size_global_kv_split = torch.split(self.size_global_kv, split_size, dim=1)
        valid_tokens_split = torch.split(self.valid_tokens, split_size, dim=1)
        cache_k_split = torch.split(self.cache_k, split_size, dim=1)
        cache_v_split = torch.split(self.cache_v, split_size, dim=1)

        for s_l_kv, s_g_kv, v_t_, cache_k, cache_v, new_cache in zip(
                size_local_kv_split, size_global_kv_split, valid_tokens_split, cache_k_split, cache_v_split, new_caches
        ):
            new_cache._seen_tokens = self._seen_tokens
            new_cache.sliding_queue_tail = self.sliding_queue_tail
            new_cache.size_local_kv = s_l_kv
            new_cache.size_global_kv = s_g_kv

            new_cache.n_ctx_local = s_l_kv
            new_cache._n_global_min = torch.min(s_g_kv)
            new_cache.n_ctx_data = torch.max(s_l_kv + s_g_kv)
            new_cache.valid_tokens = v_t_
            new_cache.chunk_fill_size = self.chunk_fill_size
            new_cache.chunk_fill_size_last = self.chunk_fill_size_last
            new_cache.cache_k = cache_k[:, :, :new_cache.n_ctx_data]
            new_cache.cache_v = cache_v[:, :, :new_cache.n_ctx_data]
        return new_caches

    def reset_cache(self, ):
        self.size_local_kv.zero_()
        self.size_global_kv.fill_(self.sliding_window_size)
        self.sliding_queue_tail = 0

        self.chunk_fill_size = 0
        self.chunk_fill_size_last = 0
        # this value is used to store the intermediate mean values from the input features
        self.cumulative_x_value = None

        self.n_ctx_local = torch.max(self.size_local_kv)
        # self.n_ctx_global = torch.max(self.size_global_kv)
        self.new_token_states = None

        self.n_ctx_data = torch.max(self.size_global_kv + self.size_local_kv)
        self._tokens_info = None
        self._end_seqs_local = None
        self._n_global_min = torch.min(self.size_global_kv)
        self.valid_tokens = None

        self.cache_k = None
        self.cache_v = None

        self._cache_k_new = None
        self._cache_v_new = None
        self._sliding_window_tokens = None

        self._seen_tokens = 0

    def update_seqs_size_info(self, key_states: torch.Tensor):
        bsz, nheads, x_len = key_states.shape[:3]
        self.n_ctx_local = torch.max(self.size_local_kv) + x_len
        # self.n_ctx_global = torch.max(self.size_global_kv) + x_len

        # self._end_seqs_local = end_seqs_info

    def get_local_fill_idx(self, x_len: int):
        local_fill_idx = torch.arange(x_len, device=self.size_local_kv.device).view(1, 1, -1)
        local_fill_idx = self.size_local_kv.unsqueeze(-1) + local_fill_idx
        return local_fill_idx

    def generate_mask(self, x_len: int):
        mask = self._generate_mask(
            size_local_kv=self.size_local_kv,
            size_global_kv=self.size_global_kv,
            n_ctx_local=self.n_ctx_local - x_len,  # type: ignore
            x_len=x_len,
        )
        return mask

    def get_seq_length(self):
        return self._seen_tokens

    def get_chunk_fill_size(self):
        return self.chunk_fill_size

    def update(self,
               key_states: torch.Tensor,
               value_states: torch.Tensor,
               token_states_info: torch.Tensor,
               ):
        """
        During update, we first consider the case where only one new token is provided. In this case, the new sliding
        tokens are added to the head of the global caches, and we can gradually remove the oldest tokens in the queue.
        However, if the number of tokens is greater than one, we first consider both local tokens and the sliding window
        tokens as local tokens to construct the attention maps and compute the attention output, we then move all the
        surviving sliding window tokens to the head of the global caches and do merging for local and sliding window
        tokens
        Args:
            key_states: torch.Tensor with shape [B, H, N_CTX, N_HEADS]
             new key states
            value_states: torch.Tensor with shape [B, H, N_CTX, N_HEADS]
             new value states
            token_states_info: torch.Tensor with shape [B, H, N_CTX, 3]
             token information, each element is a one hot encoded model. 0 indicates global tokens, 1 indicates local
             tokens, 2 indicates sliding window tokens.

        Returns:

        """
        bsz, nheads, x_len, head_dim = key_states.shape
        self.chunk_fill_size += x_len
        if token_states_info is None:
            # normally this should not happen
            assert self.chunk_fill_size < self.chunk_size, "a new token_states_info must be generated if the " \
                                                           "current cached value goes beyond the block size"
        if key_states.shape[2] == 0:
            return self.cache_k[:, :, :self.n_ctx_data], self.cache_v[:, :, :self.n_ctx_data]
        self._tokens_info = token_states_info

        # used for moving these values to the head of tokens in the end
        # self._cache_k_new = key_states[:, :, -self.sliding_window_size:]
        # self._cache_v_new = value_states[:, :, -self.sliding_window_size:]

        self._sliding_window_tokens = None if token_states_info is None else token_states_info[..., 2]

        # now we would like to move the end_seqs_local
        bsz, nheads, x_len, head_dim = key_states.shape
        self._end_seqs_local = None if token_states_info is None else token_states_info[..., 0].clone()

        if self.cache_k is None:
            # self._end_seqs_local = token_states_info[..., 0].clone()
            sliding_window_tokens = torch.zeros(
                [*key_states.shape[:2], self.sliding_window_size, key_states.shape[-1]],
                device=key_states.device, dtype=key_states.dtype
            )
            pad_token_size = get_pad_size(x_len, self.chunk_size)
            pad_token = torch.zeros(
                [*key_states.shape[:2], pad_token_size, key_states.shape[-1]],
                device=key_states.device, dtype=key_states.dtype
            )
            self.cache_k = torch.cat([sliding_window_tokens, key_states, pad_token], dim=-2)
            self.cache_v = torch.cat([sliding_window_tokens, value_states, pad_token], dim=-2)

            self.valid_tokens = torch.zeros(
                [*token_states_info.shape[:2], self.sliding_window_size],
                device=token_states_info.device, dtype=torch.bool,
            )
            self.n_ctx_local = key_states.shape[2]
            self.valid_tokens = F.pad(self.valid_tokens, (0, x_len + pad_token_size), value=1)
            if pad_token_size > 0:
                self.valid_tokens[:, :, -pad_token_size:] = False
            return key_states, value_states
        # Otherwise, we attach key values to self.cache_key
        self.update_seqs_size_info(key_states)

        # if x_len == 1:
        # in this case, we do a quick forward pass
        cache_size = self.cache_k.shape[2]
        fill_start = self.n_ctx_data
        score_size = fill_start + x_len
        n_next_chunk = math.ceil(score_size / self.chunk_size) * self.chunk_size
        pad_size = n_next_chunk - cache_size
        if pad_size > 0:
            self.cache_k = F.pad(self.cache_k, (0, 0, 0, pad_size))
            self.cache_v = F.pad(self.cache_v, (0, 0, 0, pad_size))
            self.valid_tokens = F.pad(self.valid_tokens, (0, pad_size), value=0)

        self.cache_k[:, :, fill_start:score_size] = key_states.to(self.cache_k)
        self.cache_v[:, :, fill_start:score_size] = value_states.to(self.cache_v)
        self.valid_tokens[:, :, fill_start:score_size] = 1
        return self.cache_k[:, :, :score_size], self.cache_v[:, :, :score_size]

    def post_update(self, x_len: int):
        # with torch.cuda.stream(mem_stream):
        self._post_update(x_len)

    def _post_update(self, x_len):
        self._seen_tokens = self._seen_tokens + x_len
        if x_len == 0 or self.chunk_fill_size < self.chunk_size:
            # if we are still in one chunk, then there is no need to further process the data
            self.chunk_fill_size_last = self.chunk_fill_size
            self.n_ctx_data += x_len
            return
        # this is the number of tokens from the last chunk, we will keep it unchanged as its chunk is still not finished
        n_tokens_last_chunk = self.chunk_fill_size % self.chunk_size
        n_new_chunks = self.chunk_fill_size // self.chunk_size

        self.move_sliding_window_tokens_to_head(n_new_chunks,
                                                n_tokens_last_chunk,
                                                x_len,
                                                # self.chunk_fill_size_last,
                                                # n_tokens_last_chunk,
                                                )
        # update the chunk fill size to fit the current states
        self.chunk_fill_size = self.chunk_fill_size % self.chunk_size
        self.chunk_fill_size_last = self.chunk_fill_size

        if self._tokens_info[..., 2].all():
            # all tokens are moved to the sliding window tokens, we only need to reset the n_ctx_data here as the new
            # tokens go to the sliding window token regions. This does not change the global and local kv size but the
            # sw token
            self.n_ctx_data = torch.max(self.size_global_kv + self.size_local_kv) + self.chunk_fill_size
            return
        if x_len == 1:
            new_chunk_start = self.n_ctx_data // self.chunk_size * self.chunk_size
            new_chunk_end = new_chunk_start + self.chunk_size
            new_k = self.cache_k[:, :, new_chunk_start:new_chunk_end]
            new_v = self.cache_v[:, :, new_chunk_start:new_chunk_end]

            is_global_tokens = (self._tokens_info[..., 0, 0] == 1.)
            # is_global_tokens = (self._tokens_info[..., 0, 0] == 1.).to(self.size_global_kv)
            offs_chunks = torch.arange(self.chunk_size, device=self.cache_k.device).view(1, 1, -1)
            new_idx = torch.where(is_global_tokens, self.size_global_kv, self.size_global_kv + self.size_local_kv)
            new_idx = new_idx.unsqueeze(-1) + offs_chunks

            """
            self.valid_tokens.scatter_(2, new_idx,
                                       0.
                                       # torch.lerp(
                                       #    torch.full(is_global_tokens.shape, fill_value=-torch.inf, device=self.valid_tokens.device, dtype=self.valid_tokens.dtype),
                                       #    torch.zeros(is_global_tokens.shape, device=self.valid_tokens.device, dtype=self.valid_tokens.dtype),
                                       #    is_global_tokens.to(self.valid_tokens)).unsqueeze(-1)
                                       )
            """

            # self.valid_tokens = trace_()

            # new_idx = (self.size_global_kv * is_global_tokens + (self.size_global_kv + self.size_local_kv) * (1-is_global_tokens)).unsqueeze(-1)

            new_idx = self.expand_idx_for_cache_values(new_idx, head_dim=new_k.shape[-1])

            self.cache_k.scatter_(2, new_idx, new_k)
            self.cache_v.scatter_(2, new_idx, new_v)

            # size_update_local = torch.where(is_global_tokens, self.size_local_kv + 1, 0)
            is_global_tokens = is_global_tokens.to(self.size_global_kv)
            size_update_local = (self.size_local_kv // self.chunk_size + 1) * is_global_tokens

            # for sliding window tokens, this vlaue is 1
            size_update_local = size_update_local + self._tokens_info[..., -1, 2].long()

            # size_update_global = torch.where(is_global_tokens, 1, 0)
            size_update_global = is_global_tokens
            self.size_global_kv += size_update_global * self.chunk_size
            self.size_local_kv -= size_update_local * self.chunk_size
        else:
            updated_cache_info = self.get_updated_cache_info(
                tokens_info=self._tokens_info,
                x_len=x_len,
                size_local=self.size_local_kv,
                size_global=self.size_global_kv,
                n_tokens_last_chunk=n_tokens_last_chunk,
            )

            self.cache_k, self.cache_v = merge_local_to_global_caches(
                [self.cache_k, self.cache_v],
                updated_cache_info,
                self.chunk_size
            )

            size_update_global = 0
            if updated_cache_info is not None:
                # self.valid_tokens = update_valid_tokens(
                #    self.valid_tokens,
                #    updated_cache_info=updated_cache_info,
                # )
                size_update_global = updated_cache_info.size_update_global

                self.size_global_kv += size_update_global
                self.size_local_kv -= updated_cache_info.size_update_local
        self.size_local_kv += n_new_chunks * self.chunk_size
        assert not (self.size_local_kv < 0).any()
        self.n_ctx_data = torch.max(self.size_global_kv + self.size_local_kv) + self.chunk_fill_size

        # the last chunk needs to be zeroed out
        self.cache_k[:, :, self.n_ctx_data:] = 0
        self.cache_v[:, :, self.n_ctx_data:] = 0

        self.valid_tokens = self.mask_invalid_tokens(
            self.valid_tokens, self.n_ctx_data,
            self.size_global_kv,
            self.size_local_kv,
            self.chunk_fill_size,
            size_update_global
        )
        self._n_global_min = torch.min(self.size_global_kv)

        if self._seen_tokens == x_len:
            # this helps to reduce the unnecessary memory within pre-filling
            n_remains = math.ceil(self.n_ctx_data / self.chunk_size) * self.chunk_size
            self.cache_k = self.cache_k[:, :, :n_remains]
            self.cache_v = self.cache_v[:, :, :n_remains]
            self.valid_tokens = self.valid_tokens[:, :, :n_remains]

            torch.cuda.empty_cache()

        self._cache_k_new = None
        self._cache_v_new = None
        self._sliding_window_tokens = None
        self._end_seqs_local = None

        # size_cache = torch.max(self.size_global_kv + self.size_local_kv)
        # self.cache_k = self.cache_k[:,:,:size_cache]
        # self.cache_v = self.cache_v[:,:,:size_cache]

    def move_sliding_window_tokens_to_head(self,
                                           n_chunk_to_update: int,
                                           n_tokens_remains: int,
                                           x_len: int,
                                           # n_tokens_in_first_chunk: int = 0,
                                           # n_tokens_last_chunk: int = 0,
                                           ):
        # we first move all the sliding windows related tokens towards the head of the global tokens:
        # the number of tokens in the complete chunks
        n_updated_chunks = min(n_chunk_to_update, self.n_chunks_per_sw)
        n_fills = n_updated_chunks * self.chunk_size

        if self._seen_tokens == x_len:
            fill_indices = torch.arange(n_fills, device=self.device)
            sliding_queue_tail = n_fills % self.sliding_window_size
        else:
            fill_indices = torch.arange(n_fills, device=self.device) + self.sliding_queue_tail
            fill_indices = fill_indices % self.sliding_window_size
            # we also need to move the tail back to the start of the chunk
            sliding_queue_tail = (self.sliding_queue_tail + n_fills) % self.sliding_window_size
            # in this case, the tail does not change

        self.sliding_queue_tail = sliding_queue_tail

        # self.valid_tokens[:, :, fill_indices] = torch.where(
        #    self._tokens_info[..., -x_sliding_window_tokens:, 2] == 1., 0, -torch.inf
        # ).to(self.valid_tokens)
        # now we want to update the valid tokens, we first repeat the self._tokens_info
        if n_tokens_remains > 0:
            # in this case, we need to remove the last chunk
            is_sw_tokens = self._tokens_info[..., -n_updated_chunks - 1:-1, 2]
        else:
            is_sw_tokens = self._tokens_info[..., -n_updated_chunks:, 2]

        is_sw_tokens_update = is_sw_tokens.repeat_interleave(self.chunk_size, -1).to(self.valid_tokens)

        sw_start = self.n_ctx_data // self.chunk_size + (n_chunk_to_update - n_updated_chunks)
        sw_start = sw_start * self.chunk_size
        sw_end = sw_start + n_fills

        self.cache_k[:, :, fill_indices] = torch.where(
            is_sw_tokens_update.unsqueeze(-1), self.cache_k[:, :, sw_start:sw_end], 0
        )
        self.cache_v[:, :, fill_indices] = torch.where(
            is_sw_tokens_update.unsqueeze(-1), self.cache_v[:, :, sw_start:sw_end], 0
        )
        self.valid_tokens[:, :, fill_indices] = is_sw_tokens_update

    def expand_idx_for_cache_values(self, idx_tensor: torch.Tensor, head_dim: int, ):
        idx_tensor = idx_tensor.unsqueeze(-1)
        idx_tensor = idx_tensor.expand(-1, -1, -1, head_dim)
        return idx_tensor

    def _generate_mask(self,
                       size_local_kv: torch.Tensor,
                       size_global_kv: torch.Tensor,
                       n_ctx_local: int,
                       x_len: int,
                       ):
        """
        Generate masks based on the end_seqs_local and valid_tokens information
        """
        if x_len == 1:
            return self._generate_mask_one_step(size_local_kv + size_global_kv)
        # here, local mask is the new tokens + tokens from the last iteration that is still stored in the chunk
        msk_new_tokens, idx_sub_seq = self._generate_local_mask(size_local_kv=size_local_kv.unsqueeze(-1),
                                                                x_len=x_len,
                                                                chunk_fill_size=self.chunk_fill_size)
        if self._seen_tokens == 0:
            return msk_new_tokens

        if self.valid_tokens.shape[-1] == 0:
            return msk_new_tokens

        # this is the number of tokens in the full chunks from
        n_tokens_in_full_chunks = self.n_ctx_data - self.chunk_fill_size + x_len

        # now we want to construct mask for cache values. This is composed of two parts, we first fit the sliding
        # window tokens
        msk_cache = self.valid_tokens[:, :, :n_tokens_in_full_chunks].unsqueeze(-2)

        msk_cache = msk_cache.repeat(1, 1, x_len, 1)
        msk_cache = self._fit_sliding_queue_masks(msk_cache, x_len, self.chunk_fill_size)

        # we now fit the local tokens. These local tokens might only survive until the first global token appear.
        # Assuming that the first global token appears at step 3 and we have x_len=5, then only 3 out of 5 tokens are
        # valid tokens
        # msk_local = msk_cache[:,:,:,self._n_global_min:self.n_ctx_data] # [bsz, nheads, x_len, n_local_max]
        # local_idx_start = (self.chunk_fill_size - x_len) // self.chunk_size
        local_idx_end = math.ceil(self.chunk_fill_size / self.chunk_size)
        msk_local = torch.arange(local_idx_end, device=idx_sub_seq.device, dtype=idx_sub_seq.dtype).view(1, 1, -1)
        msk_local = (msk_local <= idx_sub_seq[:, :, [0]]).unsqueeze(-1)  # [bsz, nheads, x_len, 1]
        is_local_tokens = torch.arange((n_tokens_in_full_chunks - self._n_global_min) // self.chunk_size,
                                       device=msk_local.device).view(1, 1, 1, -1)

        local_tokens_upper = (self.size_local_kv + self.size_global_kv - self._n_global_min)[
                                 ..., None, None] // self.chunk_size
        local_tokens_lower = (self.size_global_kv - self._n_global_min)[..., None, None] // self.chunk_size
        msk_local = torch.where(local_tokens_lower <= is_local_tokens, msk_local, True)
        msk_local = torch.where(is_local_tokens < local_tokens_upper, msk_local, False)

        # now we remap the chunks to the raw size
        msk_local = msk_local.repeat_interleave(self.chunk_size, dim=-1).repeat_interleave(self.chunk_size, dim=-2)
        msk_local = msk_local[:, :, self.chunk_fill_size - x_len:self.chunk_fill_size, :]

        """
        msk_local = torch.arange(self.chunk_fill_size - x_len, self.chunk_fill_size, device=idx_sub_seq.device, dtype=idx_sub_seq.dtype).view(1, 1, -1)
        msk_local = (msk_local <= idx_sub_seq[:, :, [0]] * self.chunk_size).unsqueeze(-1)  # [bsz, nheads, x_len, 1]
        is_local_tokens = torch.arange(n_tokens_in_full_chunks - self._n_global_min, device=msk_local.device).view(1, 1, 1, -1)
        local_tokens_upper = (self.size_local_kv + self.size_global_kv - self._n_global_min)[..., None, None]
        local_token_lower = (self.size_global_kv - self._n_global_min)[..., None, None]

        # is_local_tokens = (local_token_lower <= is_local_tokens < local_tokens_upper).unsqueeze(-2)
        msk_local = torch.where(local_token_lower <= is_local_tokens, msk_local, True)
        msk_local = torch.where(is_local_tokens < local_tokens_upper, msk_local, False)
        """

        # This is only applied to local masks

        msk_cache[:, :, :, self._n_global_min:n_tokens_in_full_chunks] = msk_local.to(msk_cache)
        return torch.cat([msk_cache, msk_new_tokens], dim=-1)

        """

        len_local = n_ctx_local + x_len
        # mask_global = self.valid_tokens[:, :, :n_ctx_global].unsqueeze(-2)
        # for the following codes, we fill the local masks to the global masks.
        # we first extract a pre-allocated global mask

        n_required_seq = self.n_ctx_data + x_len
        n_global_max = self.size_global_kv.max()  # TODO this should be updated further!
        # the placeholder local size is the one element larger than the required msk tensors, the addition place is used
        # to store the invalid mask values

        mask_global = F.pad(
            # self.valid_tokens, (0, n_required_seq + 1 - self.valid_tokens.shape[2]), mode='constant', value=-torch.inf
            self.valid_tokens[:, :, :n_global_max], (0, n_required_seq + 1 - n_global_max), mode='constant',
            value=-torch.inf
        ).unsqueeze(-2)

        if x_len > 1:
            # mask_global = mask_global.expand(-1, -1, local_mask.shape[-2], -1)
            mask_global = mask_global.repeat(1, 1, local_mask.shape[-2], 1)
            mask_global = self._fit_sliding_queue_masks(mask_global, x_len)

        device = mask_global.device
        # computing the required index
        index_local_msk = torch.arange(len_local, device=device).view(1, 1, -1)

        index_local_msk = index_local_msk + size_local_kv.unsqueeze(-1) - n_ctx_local
        # index_local_msk < 0 are invalid masks, we place them at the end of the mask_global tokens and remove
        # that later on
        index_local_msk = torch.where(index_local_msk >= 0,
                                      size_global_kv.unsqueeze(-1) + index_local_msk,
                                      mask_global.shape[-1] - 1,
                                      )

        index_local_msk = index_local_msk.unsqueeze(-2).expand(-1, -1, local_mask.shape[-2], -1)

        # mask is then assigned to corresponding the global values
        mask = mask_global.scatter_(-1, index_local_msk, local_mask.to(mask_global))
        mask = mask[:, :, :, :self.n_ctx_data + x_len]

        mask = mask.to(size_local_kv.device)

        return mask
        """

    @torch.compile
    def generate_block_sparse_mask_info_one_step(self):
        """
        This function generate a block sparse mask to efficiently skip some of the blocks that are no longer valid. In
        NAtS, we always first place global tokens, then local tokens and then in valid tokens,
        Returns:
            indices: block indices of the block sparse matrix
            indices: the valid blocks, in nats, we always assume that the invalid tokens are located
        """
        # TODO this function is not finished yet, we need the offsets for each heads as we flatten the input tensors!!!
        n_blocks = ((self.size_local_kv + self.size_global_kv) // self.chunk_size).flatten().to(torch.int32)
        # n_new_chunk = math.ceil(self.chunk_fill_size / self.chunk_size)
        n_new_chunk = 1
        # for one step, we can ensure that there is only one new chunk
        n_blocks_valid = n_blocks + n_new_chunk
        n_last_block = torch.max(n_blocks_valid)
        indptr = F.pad(torch.cumsum(n_blocks_valid, dim=-1), (1, 0))
        indices = torch.zeros(indptr[-1])
        for i in range(len(indptr) - 1):
            filled_value = torch.arange(n_blocks[i], dtype=torch.int32, device=self.size_local_kv.device)
            # in practice, if we are applying different masks for each head to flashinfer, we need to
            # view q from (h_q,1,d) to (h_k, n*h_k, d) and flatten k and v from (h_k, n, d)
            # to (1, n*h_k, d) such that different block sparse masks can be applied to different heads
            indices[indptr[i]: indptr[i + 1] - 1] = filled_value + n_last_block * i
            # indices[indptr[i+1] - n_new_chunk:indptr[i+1]] = torch.arange(n_blocks[i]+1, n_blocks[i]+n_new_chunk, dtype=torch.int32,device=self.size_local_kv.device)
            indices[indptr[i + 1]] = n_last_block * i
        n_invalid = (self.sliding_window_size - self.valid_tokens[:, :, :self.sliding_window_size].long().sum(-1))
        n_invalid += self.chunk_size - (self.chunk_fill_size % self.chunk_size)
        return indptr, indices, n_invalid

    def _generate_local_mask(self, size_local_kv: torch.Tensor, x_len: int = 0, chunk_fill_size: int = 0) -> tuple[
        torch.Tensor, torch.Tensor]:
        # end_seqs_local = F.pad(self._end_seqs_local, (n_ctx_local, 0), mode='constant')

        # msk = generate_hard_seq_masks(end_seqs_local, start_idx=n_ctx_local)
        if self._end_seqs_local is None:
            end_seqs_hard = torch.ones([*size_local_kv.shape, 1], device=size_local_kv.device, dtype=torch.float16)
        else:
            end_seqs_hard = self._end_seqs_local
        msk, idx_sub_seq = generate_hard_seq_masks(end_seqs_hard=end_seqs_hard, start_idx=0)

        n_ctx_compressed = end_seqs_hard.shape[-1]

        # we now fill sliding window masks
        if self._sliding_window_tokens.any():
            n_ranges = torch.arange(n_ctx_compressed, device=self.device)
            sliding_window = n_ranges.unsqueeze(1) - n_ranges.unsqueeze(0)
            sliding_window_msk = (sliding_window <= (self.sliding_window_size // self.chunk_size))
            msk[:, :, :, -n_ctx_compressed:] = torch.where(
                self._sliding_window_tokens.unsqueeze(-2) == 1.,
                sliding_window_msk.view(1, 1, n_ctx_compressed, n_ctx_compressed),
                msk[:, :, :, -n_ctx_compressed:]
            )

        msk = msk.repeat_interleave(self.chunk_size, -1).repeat_interleave(self.chunk_size, -2)
        msk = msk[:, :, chunk_fill_size - x_len:chunk_fill_size, :chunk_fill_size]

        msk = msk.tril_(diagonal=chunk_fill_size - x_len)

        # seq_mask_ = msk
        # msk = torch.where(seq_mask_, 0.0, -torch.inf)
        return msk, idx_sub_seq

    def _generate_mask_one_step(self,
                                size_cache: torch.Tensor):
        """
        if we only need to generate mask for one single step, since we already know that all the cached values will
        contribute to the new output generation, we could directly check the size of the valid tokens
        """
        if self._seen_tokens == 0:
            return torch.ones([1], device=size_cache.device, dtype=torch.bool).view(1, 1, 1, 1)
        # mask = F.pad(self.valid_tokens, (0, 1)).unsqueeze(-2)
        mask = self.valid_tokens[:, :, :self.n_ctx_data + 1].unsqueeze(-2)
        return mask

    def mask_invalid_tokens(self, valid_tokens: torch.Tensor, n_ctx_data: int, size_global_kv: torch.Tensor,
                            size_local_kv: torch.Tensor,
                            size_in_chunk: int,
                            size_update_local: torch.Tensor | int):
        # if n_ctx_data >= valid_tokens.shape[2]:
        #    pad_size = n_ctx_data - valid_tokens.shape[2]
        #    valid_tokens = F.pad(
        #        valid_tokens, (0, pad_size), mode='constant', value=-torch.inf)

        # size_global_min = self._n_global_min or (size_global_kv - size_update_local).min()
        size_global_min = self._n_global_min
        if n_ctx_data == size_global_min:
            return valid_tokens

        n_last_chunk = n_ctx_data // self.chunk_size * self.chunk_size
        involved_idx_range = (torch.arange(n_last_chunk - size_global_min, device=valid_tokens.device)).view(1, 1, -1)
        n_last = (size_global_kv + size_local_kv - size_global_min).unsqueeze(-1)
        fill_value = involved_idx_range < n_last
        valid_tokens[:, :, size_global_min:n_last_chunk] = fill_value
        # used for
        valid_tokens[:, :, n_last_chunk:n_last_chunk + size_in_chunk] = True

        return valid_tokens

    def _fit_sliding_queue_masks(self,
                                 mask_global: torch.Tensor,
                                 x_len: int,
                                 chunk_fill_size: int
                                 ):
        # we also need to check the maks for sliding windows, in this case, only the first self.sliding_window_size
        # have the valid masks
        # in this case, the sliding window queue is still not full: the head of the queue is 0
        x_len_range = (torch.arange(chunk_fill_size - x_len, chunk_fill_size, device=self.device).unsqueeze(
            1) + self.chunk_fill_size_last) // self.chunk_size
        seen_chunks = self._seen_tokens // self.chunk_size
        if self._seen_tokens < self.sliding_window_size:
            queue_range = torch.arange(seen_chunks, device=self.device).unsqueeze(0)
            queue_range = queue_range.repeat_interleave(self.chunk_size, 1)
            # mask_global[..., :self._seen_tokens] = torch.where(
            #    (x_len_range + queue_range) < self.sliding_window_size, 0., -torch.inf
            # ).view(1, 1, x_len, self._seen_tokens)

            msk_sw = ((x_len_range + queue_range) < self.n_chunks_per_sw).view(
                1, 1, -1, seen_chunks * self.chunk_size,
            )
            mask_global[..., : seen_chunks * self.chunk_size] &= msk_sw

        else:
            # in other case, the queue is full, the head of the queue is (tail+1) % sliding_window_size
            queue_range = torch.arange(self.n_chunks_per_sw, device=self.device) * self.chunk_size + self.chunk_size
            queue_range = queue_range.repeat_interleave(self.chunk_size, 0)
            queue_range = (
                                      queue_range + self.sliding_window_size - 1 - self.sliding_queue_tail) % self.sliding_window_size
            if chunk_fill_size <= self.sliding_window_size:
                # msk_global_update = torch.full(
                #    (x_len, self.sliding_window_size), fill_value=-torch.inf,
                #    device=mask_global.device, dtype=mask_global.dtype
                # ).tril(-1)

                msk_global_update = torch.ones(
                    (chunk_fill_size, self.sliding_window_size), device=mask_global.device, dtype=mask_global.dtype
                ).triu(0)
                mask_global_update = msk_global_update[chunk_fill_size - x_len:chunk_fill_size, queue_range]
            else:
                # msk_global_update = torch.full(
                #    (self.sliding_window_size, self.sliding_window_size),
                #    fill_value=-torch.inf, device=mask_global.device, dtype=mask_global.dtype
                # ).tril(-1)
                msk_global_update = torch.ones(
                    (self.sliding_window_size, self.sliding_window_size), device=mask_global.device,
                    dtype=mask_global.dtype
                ).triu(0)
                msk_global_update = msk_global_update[chunk_fill_size - x_len:, queue_range]
                mask_global_update = F.pad(msk_global_update, (0, 0, 0, x_len - msk_global_update.shape[0]), )
            # for each new tokens, we need to remove the first corresponding values in the token

            mask_global[..., :self.sliding_window_size] &= mask_global_update.view(1, 1, x_len,
                                                                                   self.sliding_window_size)

        return mask_global

    def get_updated_cache_info(self,
                               tokens_info: torch.Tensor,
                               size_local: torch.Tensor,
                               size_global: torch.Tensor,
                               x_len: int,
                               n_tokens_last_chunk: int,
                               ) -> CacheUpdateInfo | None:
        """
        This function is provided to check if any information from the new sequence (including the cached local
        sequence and the new sequence) can be compressed into the global sequence.
        For the others, we simply put the corresponding values to the tail of each sequences

        Given that every item from end_seqs_local must contain at least one sequence with only 0. we could easily check
        which item in the batch will be updated by comparing the concatenated new sequence and the old local sequence.
        Once we have found which sample in the batch needs to be updated, we could extract the tokens that should be
        placed under the global caches.


        Args:
            tokens_info (torch.Tensor) torch Tensor with size (bsz, nh, x_len,n_states), the states of the new tokens
            size_local (torch.Tensor) torch Tensor with size (bsz, nh), size of the existing local tensors, this
                value is applied to compute the required reduced size for local kv models
            x_len (torch.Tensor): length of the new input sequences

        Returns:
            updated_cache_info (dict[int, torch.Tensor]): an updated cache information. This item is used to show
                which token in the local and new sequences need to be moved to the global sequence.

        """
        # the existing local end_seqs must contain at least one sequence that only contain 0
        # Hence, if the new end_seqs need to be merged to the global cache, then none of its subsequence should contain
        # a pure 0 sequence
        if tokens_info[..., 1].all():
            # all the tokens are local tokens, there is no need to do the update
            return None
        if tokens_info[..., 2].all():
            # all the tokens are sliding window tokens, we only need to take care of the sliding window tokens
            return CacheUpdateInfo(b_idx=[],
                                   h_idx=[],
                                   update_idx=[],
                                   shift_idx=[],
                                   seq_len=x_len,
                                   valid_value_mask=None,
                                   size_update_global=torch.zeros_like(size_global),
                                   size_update_local=torch.ones_like(size_global) * x_len,
                                   global_token_masks=None,
                                   )

        if x_len == 1:
            return self.get_updated_cache_info_one_step(tokens_info, size_local, size_global)
        global_tokens = tokens_info[..., 0]
        bsz, n_heads, seq_len = global_tokens.shape

        # now we need to determine the end tokens that we would like to put into the global tokens. Since the remaining
        # local tokens must contain a sequence with only 0. Hence, we check the tokens that result in a shortest seq
        # that only contains 0. i.e., the position of the token where the last True in that sequence appears
        # value_ranges = torch.where(end_seqs_local[idx2update], torch.arange(end_seqs_local.shape[-2]).unsqueeze(-1), 0.0
        # ).max(1)[0].min(-1)[0].long()

        # the number of tokens added to the global tokens
        size_update_global = global_tokens.long().sum(-1)

        global_token_indices = torch.where(
            global_tokens > 0.,
            torch.arange(seq_len, device=global_tokens.device).view(1, 1, -1),
            -1
        )
        if n_tokens_last_chunk != 0:
            # in this case, the last global token will be incomplete,
            size_update_global = size_update_global - 1
            global_token_indices[..., -1] = -1
        size_update_global = size_update_global * self.chunk_size

        last_global_idx: torch.Tensor = global_token_indices.max(-1)[0]

        msk_removed_tokens = torch.arange(seq_len, device=tokens_info.device).view(1, 1, -1)
        msk_removed_tokens = msk_removed_tokens > last_global_idx.unsqueeze(-1)

        size_update_local = torch.where(last_global_idx > -1, last_global_idx * self.chunk_size + size_local, 0).long()

        remaining_local_tokens = torch.where(msk_removed_tokens, tokens_info[..., 1], 0).long()

        n_sliding_window_tokens = seq_len - F.relu(last_global_idx) - remaining_local_tokens.sum(-1)
        if n_tokens_last_chunk != 0:
            # remove the information from the unfinished token
            n_sliding_window_tokens -= 1
        # sliding window tokens will be removed anyway
        size_update_local += n_sliding_window_tokens * self.chunk_size

        # these are the tokens that need to be preserved in the cache
        remaining_tokens = (global_tokens + remaining_local_tokens).long()

        last_chunk_needs_move = n_tokens_last_chunk != 0

        if last_chunk_needs_move:
            # in this case, the remaining tokens should be ignored as they are still incomplete
            remaining_tokens[:, :, -1] = 0

        non_zero_idx = torch.nonzero(remaining_tokens)
        # provide indexing information for each element
        # non_zero_seqs = non_zero_idx[:, 0] * n_heads + non_zero_idx[:, 1]
        # since non_zero_seqs is already sorted, we could check the number of each elements
        # counts_each_seq = torch.unique_consecutive(non_zero_seqs, return_counts=True)[1]
        counts_each_seq = remaining_tokens.sum(-1).flatten()
        max_update_size = counts_each_seq.max()
        # the following codes is equivalent to torch.cat([torch.arange(x) for x in max_update_size])
        # TODO check which function is faster!
        max_update_size_ranges = torch.arange(max_update_size, device=counts_each_seq.device)
        mask = (max_update_size_ranges.unsqueeze(0) < counts_each_seq.unsqueeze(1))

        shift_idx = max_update_size_ranges.repeat(len(counts_each_seq))[mask.flatten()]

        sizes_all = (size_global + size_local) // self.chunk_size
        new_model_base = torch.where(last_global_idx == -1, sizes_all, size_global // self.chunk_size)

        b_idx = non_zero_idx[:, 0]
        h_idx = non_zero_idx[:, 1]

        # we also need to know which tokens are global tokens and which not
        global_token_masks = non_zero_idx[:, 2] <= last_global_idx[b_idx, h_idx]

        shift_idx += new_model_base[b_idx, h_idx]
        # update_idx = non_zero_idx[:, 2] + sizes_all[b_idx, h_idx]
        update_idx = non_zero_idx[:, 2] + self.n_ctx_data // self.chunk_size

        if last_chunk_needs_move and (max_update_size < (remaining_tokens.shape[-1] - 1)):
            # now we want to move the remaining tokens to the end of the longest sequence

            b_idx = torch.cat([b_idx, torch.arange(bsz, device=b_idx.device).repeat_interleave(n_heads)])
            h_idx = torch.cat([h_idx, torch.arange(n_heads, device=h_idx.device).repeat(bsz)])
            # the remaining incomplete tokens must be located in the end of the new sequence
            update_idx_incomplete = torch.full(
                [bsz * n_heads], fill_value=remaining_tokens.shape[-1] - 1 + self.n_ctx_data // self.chunk_size,
                dtype=update_idx.dtype, device=update_idx.device,
            )
            # Its shift idx
            update_idx = torch.cat([update_idx, update_idx_incomplete])

            shift_idx_incomplete = torch.full(
                [bsz * n_heads], fill_value=torch.max(counts_each_seq.view(bsz, n_heads) + new_model_base),
                device=shift_idx.device, dtype=shift_idx.dtype
            )
            shift_idx = torch.cat([shift_idx, shift_idx_incomplete])

        return CacheUpdateInfo(b_idx=b_idx,
                               h_idx=h_idx,
                               update_idx=update_idx,
                               shift_idx=shift_idx,
                               seq_len=seq_len,
                               valid_value_mask=None,
                               size_update_global=size_update_global,
                               size_update_local=size_update_local,
                               global_token_masks=global_token_masks,
                               )

    def get_updated_cache_info_one_step(self,
                                        tokens_info: torch.Tensor,
                                        size_local: torch.Tensor,
                                        size_global: torch.Tensor,
                                        ) -> CacheUpdateInfo | None:
        """
        If only one new value is updated each time, we can ensure that each sequence in end_seqs_local only contains one
        True value, therefore, we only need to check if end_seqs_local contains a True value and collect those
        """
        seq_len = 1

        b_idx, h_idx, update_idx = torch.nonzero(tokens_info[..., 0], as_tuple=True)
        size_update_global = torch.zeros_like(size_local)
        size_update_global[b_idx, h_idx] = 1

        size_update_local = torch.zeros_like(size_local)

        size_local_involved = size_local[b_idx, h_idx]
        size_global_involved = size_global[b_idx, h_idx]

        size_update_local[b_idx, h_idx] = update_idx + 1 + size_local_involved
        # we also need to remove the sliding window tokens
        size_update_local += tokens_info[:, :, -1, 2].long()

        return CacheUpdateInfo(
            b_idx=b_idx,
            h_idx=h_idx,
            update_idx=update_idx + size_local_involved + size_global_involved,
            shift_idx=size_global_involved,
            seq_len=seq_len,
            # b_idx_unique=b_idx,
            # h_idx_unique=h_idx,
            valid_value_mask=None,
            size_update_global=size_update_global,
            size_update_local=size_update_local,
            global_token_masks=None
        )
