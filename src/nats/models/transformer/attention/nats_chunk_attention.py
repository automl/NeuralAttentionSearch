import math

import torch
from torch import nn
import torch.nn.functional as F

from nats.models.transformer.attention.nats_attention import (
    BaseNAtSAttentionLayer,
    GPTAttentionLayer,
    LLAMA3AttentionLayer
)
from nats.components.cache.dyn_cache import TransformerCache, NAtSCache
from nats.models.transformer.utils import repeat_masks
from nats.chunk_utils import ChunkMergeType, ChunkMergeFuncs, get_pad_size

from nats.models.transformer.triton.nats_chunk_attention import nats_chunk_attention


class BaseNAtSChunkAttentionLayer(BaseNAtSAttentionLayer):
    """
    This is an experimental segment attention that incoporate the end_seq information directly into the attention
    computations
    """

    def __init__(self,
                 dim: int = 4096,
                 n_heads: int = 32,
                 n_kv_heads: int | None = None,
                 layer_id: int = 0,
                 dropout: float = 0.0,
                 use_flash: bool = True,
                 on_ddp: bool = False,
                 n_msks: int | None = None,
                 sparse_regularized_value: float = 0.0,
                 local_seq_max_length: int = 4,
                 n_options: int = 3,
                 chunk_size: int = 8,
                 chunk_merge_method: str = 'mean',
                 compress_on_q: bool = False,
                 ):
        super(BaseNAtSChunkAttentionLayer, self).__init__(dim, n_heads, n_kv_heads, layer_id, dropout, use_flash,
                                                          on_ddp, n_msks,
                                                          sparse_regularized_value, local_seq_max_length, n_options,
                                                          )
        if chunk_merge_method == 'mean':
            self.chunk_merge_method = ChunkMergeType.MEAN
        elif chunk_merge_method == 'max':
            self.chunk_merge_method = ChunkMergeType.MAX
        else:
            raise ValueError(f'Unknown chunk merge method: {chunk_merge_method}')
        self.chunk_merge_func = ChunkMergeFuncs[self.chunk_merge_method]
        self.chunk_size = chunk_size
        self.compress_on_q = compress_on_q

    def get_pad_size(self, x_len):
        return math.ceil(x_len / self.chunk_size) * self.chunk_size - x_len

    def _sample_for_token_info_values(self, x: torch.Tensor, ):
        n_ctx = x.shape[1]
        pad_size = get_pad_size(n_ctx, self.chunk_size)
        x_subsample = self.chunk_merge_func(x, self.chunk_size, pad_size, )
        return super()._sample_for_token_info_values(x_subsample)

    def _sample_for_token_info_values_inference(self, x: torch.Tensor, start_pos,
                                                cache: NAtSCache | None = None) -> torch.Tensor:
        """
        The usage of this function here is two-fold. First, we generate the token info for the new x. Second, we update
        the intermediate cached x values for the cache
        Args:
            x: torch.Tensor, new input tensor of shape (bsz, nheads, x_len, n_dims)
            start_pos: int start position, we might need to do different sampling given different start pos.
                For instance,
            cache: NAtSCache, applied to provide past cached information on x

        Returns:
            samples: torch.Tensor, new samples

        """
        x_len = x.shape[1]
        pre_fill_chunk_size = cache.get_nats_chunk_fill_size(self.layer_id)
        if pre_fill_chunk_size + x_len < self.chunk_size:
            cache.update_nats_cached_x_in_chunk(x, self.layer_id, self.chunk_merge_method)

            return self.one_hot_values.to(x).reshape((1, 1, 1, -1)).expand(x.shape[0], self.n_kv_heads, 1, -1)

        x_subsample, pad_size = cache.update_nats_cached_x_across_chunk(x,
                                                                        pre_fill_chunk_size,
                                                                        self.layer_id,
                                                                        self.chunk_merge_method)
        samples = super()._sample_for_token_info_values_inference(x_subsample, start_pos, sink_token_end=self.chunk_size)

        if pad_size != 0:
            # This marks the last uncompleted chunk as a global token to make it easier for the following post-update
            samples[:, :, -1] = self.one_hot_values.to(samples)
        return samples

    def multi_head_attention(self, xq: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                             mask: torch.Tensor | None = None,
                             cache: NAtSCache | None = None,
                             is_first_module: bool = True,
                             kwargs: dict | None = None,
                             ):
        assert 'token_states_info' in kwargs, 'Token States Infor is required for multi head attention!'
        token_states_info = kwargs['token_states_info']
        if self.training:
            # end_seqs = self._sample_for_token_info_values(xv)
            end_seq_idx = self.get_end_seq_indices(token_states_info[..., 0])
            # end_seqs = repeat_end_seqs_values(end_seqs, self.n_rep_msk)
            # end_seq_idx = repeat_end_seqs_values(end_seq_idx, self.n_rep_msk)
            # here we estimate the number of valid tokens remained in our models
            N_CTX = end_seq_idx.shape[-1]
            n_value_ranges = torch.arange(N_CTX).to(token_states_info)
            n_valid_tokens = torch.where(
                token_states_info[..., 0] == 1., N_CTX - n_value_ranges, end_seq_idx - n_value_ranges
            )
            n_valid_tokens = torch.where(token_states_info[..., -1] == 1., self.local_seq_max_length // self.chunk_size,
                                         n_valid_tokens)

            self.end_seqs_size = torch.mean(n_valid_tokens / N_CTX * 2)
            res = nats_chunk_attention(
                xq.contiguous(), keys.contiguous(), values.contiguous(), token_states_info, end_seq_idx,
                causal=True, sm_scale=self.sm_scale,
                nats_chunk_size=self.chunk_size,
                n_rep=self.n_rep,
                sparse_regularized_value=self.sparse_regularized_value,
                local_seq_max_length=self.local_seq_max_length,
                compress_on_q=self.compress_on_q
            )
            return res
        else:
            x_len = xq.shape[-2]
            mask = cache.generate_mask(layer_idx=self.layer_id, x_len=x_len)
            mask = repeat_masks(mask, n_rep=self.n_rep)

            if self._use_flash:
                output = F.scaled_dot_product_attention(
                    xq, keys, values,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False,
                    # is_causal=True,
                )

            else:
                scores = torch.matmul(xq, keys.transpose(2, 3)) * (1.0 / math.sqrt(self.head_dim))
                scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                # scores = self.attn_dropout(scores)
                output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)

            if self.executor is not None:
                self._cache_is_till_updating = self.executor.submit(
                    cache.post_update, layer_idx=self.layer_id, x_len=x_len,
                )
            else:
                cache.post_update(layer_idx=self.layer_id, x_len=x_len, )
            return output


class GPTNAtSChunkAttentionLayer(BaseNAtSChunkAttentionLayer, GPTAttentionLayer):
    pass


class LlamaNAtSChunkAttentionLayer(BaseNAtSChunkAttentionLayer, LLAMA3AttentionLayer):
    pass
