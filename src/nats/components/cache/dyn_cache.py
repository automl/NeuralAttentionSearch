from typing import Any
from transformers.cache_utils import DynamicCache

import torch

from nats.chunk_utils import get_pad_size, ChunkMergeType, ChunkMergeFuncs
from nats.components.cache.transformer_cache import TransformerLayerCache
from nats.components.cache.nats_layer_cache import NAtSLayerCache
from nats.components.cache.nats_chunk_layer_cache import NAtSChunkLayerCache

from nats.components.cache.baselines import (
    HHCache,
    AttentionSinkCache,
    CAMCache,

)

CacheTypeDict = {
    'transformer': TransformerLayerCache,
    'att_sink': AttentionSinkCache,
    'h2o': HHCache,
    'cam': CAMCache
}


class TransformerCache(DynamicCache):
    def __init__(self, cache_type: str, n_kv_heads: int, cache_init_kwargs: dict):
        super(TransformerCache, self).__init__()
        self.cache_type = cache_type
        self.cache_transformer = []
        self.cache_init_kwargs = cache_init_kwargs
        self.n_kv_heads = n_kv_heads

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: dict[str, Any] | None = None,
    ):
        if len(self.cache_transformer) <= layer_idx:
            self.cache_transformer.append(CacheTypeDict[self.cache_type](**self.cache_init_kwargs))
        return super(TransformerCache, self).update(key_states, value_states, layer_idx, cache_kwargs)

    def post_update(self, layer_idx: int, **kwargs):
        if self.cache_type == 'transformer':
            return
        elif self.cache_type == 'att_sink':
            self.key_cache[layer_idx], self.value_cache[layer_idx] = self.cache_transformer[layer_idx](
                (self.key_cache[layer_idx], self.value_cache[layer_idx])
            )
        elif self.cache_type == 'cam':
            attn_weights = kwargs['attn_weights']
            self.key_cache[layer_idx], self.value_cache[layer_idx] = self.cache_transformer[layer_idx](
                attn_weights, (self.key_cache[layer_idx], self.value_cache[layer_idx])
            )
        elif self.cache_type == 'h2o':
            attn_weights = kwargs['attn_weights']
            self.key_cache[layer_idx], self.value_cache[layer_idx] = self.cache_transformer[layer_idx](
                attn_weights, (self.key_cache[layer_idx], self.value_cache[layer_idx]), self.n_kv_heads
            )
        else:
            raise NotImplementedError(f'Unknown cache type: {self.cache_type}')


class NAtSCache(DynamicCache):
    def __init__(self, sliding_window_size: int, chunk_size: int = 1):
        # model 1
        self.cache_model: list[NAtSLayerCache | NAtSChunkLayerCache] = []
        self.sliding_window_size = sliding_window_size
        self.chunk_size = chunk_size
        self.cached_x = []  # This function is used only for chunk-aware cache.
        # TODO add chunk cache for NAtS!!!

    def reset_cache(self):
        self.cache_model: list[NAtSLayerCache | NAtSChunkLayerCache] = []
        self.cached_x = []

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.cache_model) <= layer_idx:
            return 0
        return self.cache_model[layer_idx]._seen_tokens

    def generate_mask(self, layer_idx: int, x_len: int, ):
        return self.cache_model[layer_idx].generate_mask(x_len)

    def get_nats_chunk_fill_size(self, layer_idx: int = 0):
        if len(self.cache_model) <= layer_idx or self.chunk_size == 1:
            return 0
        else:
            return self.cache_model[layer_idx].chunk_fill_size

    def get_nats_cached_x_in_chunk(self, layer_idx: int = 0):
        if len(self.cached_x) <= layer_idx:
            return None
        return self.cached_x[layer_idx]

    @staticmethod
    def merge_new_x_values_in_chunk(x_cache: torch.Tensor | None,
                                    x_new: torch.Tensor,
                                    chunk_size: int,
                                    chunk_merge_method: ChunkMergeType = ChunkMergeType.MEAN):
        if x_cache is not None:
            if chunk_merge_method == ChunkMergeType.MEAN:
                x_update = x_cache + (x_new / chunk_size).sum(1, keepdim=True)
            else:
                x_update = torch.max(torch.cat([x_cache, x_new], dim=1), dim=1, keepdim=True)[0]
        else:
            if chunk_merge_method == ChunkMergeType.MEAN:
                x_update = (x_new / chunk_size).sum(1, keepdim=True)
            else:
                x_update = x_new
        return x_update

    def update_nats_cached_x_in_chunk(self, x: torch.Tensor, layer_idx: int = 0,
                                      chunk_merge_method: ChunkMergeType = ChunkMergeType.MEAN):
        """
        This function update the cached x of within one chunk
        """
        x_cache = self.get_nats_cached_x_in_chunk(layer_idx)
        x_update = self.merge_new_x_values_in_chunk(x_cache, x, self.chunk_size, chunk_merge_method)
        self._update_cache_x(x_update, layer_idx)

    def _update_cache_x(self, x_update, layer_idx: int):
        if len(self.cached_x) <= layer_idx:
            self.cached_x.append(x_update)
        else:
            self.cached_x[layer_idx] = x_update

    def update_nats_cached_x_across_chunk(self, x: torch.Tensor,
                                          pre_fill_chunk_size: int,
                                          layer_idx: int = 0,
                                          chunk_merge_method: ChunkMergeType = ChunkMergeType.MEAN,
                                          ) -> tuple[torch.Tensor, int]:
        """
        This function update the cached x values across different chunks
        """
        pre_fill_chunk_size = self.get_nats_chunk_fill_size(layer_idx)
        x_len = x.shape[1]
        if pre_fill_chunk_size != 0:
            # in this case, we might need to incorporate the information from the previous cached values
            # we first compute the token state for the first chunk
            size_in_previous_chunk = self.chunk_size - pre_fill_chunk_size
            x_start = x[:, :size_in_previous_chunk]
            x_cache = self.get_nats_cached_x_in_chunk(layer_idx)
            x_start_update = self.merge_new_x_values_in_chunk(x_cache, x_start, self.chunk_size, chunk_merge_method)
            x_len_remains = x_len - size_in_previous_chunk
            if x_len_remains == 0:
                # In this case, we only need to care about hte
                pad_size = 0
                x_subsample = x_start_update
            else:
                pad_size = get_pad_size(x_len_remains, self.chunk_size)
                x_subsample = ChunkMergeFuncs[chunk_merge_method](x[:, size_in_previous_chunk:], self.chunk_size, pad_size, )
                x_subsample = torch.cat([x_start_update, x_subsample], dim=1)
        else:
            pad_size = get_pad_size(x_len, self.chunk_size)
            x_subsample = ChunkMergeFuncs[chunk_merge_method](x, self.chunk_size, pad_size, )

        if pad_size == 0:
            # in this case, there is no need to store the intermediate x values
            x_update = None
        else:
            x_new_chunk = x[:, pad_size - self.chunk_size:]
            if chunk_merge_method == ChunkMergeType.MEAN:
                x_update = torch.sum(x_new_chunk / self.chunk_size, 1, keepdim=True)
            else:
                x_update = torch.max(x_new_chunk, 1, keepdim=True)[0]
        self._update_cache_x(x_update, layer_idx)

        return x_subsample, pad_size

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: dict[str, Any] | None = None,
    ):
        if len(self.cache_model) <= layer_idx:
            bsz, n_kv_heads = key_states.shape[:2]
            if self.chunk_size == 1:
                new_cache = NAtSLayerCache(bsz, n_kv_heads,
                                           self.sliding_window_size,
                                           layer_id=layer_idx, device=key_states.device)
            else:
                new_cache = NAtSChunkLayerCache(bsz, n_kv_heads, self.sliding_window_size,
                                                layer_id=layer_idx, chunk_size=self.chunk_size,
                                                device=key_states.device)
            self.cache_model.append(new_cache)
        key, value = self.cache_model[layer_idx].update(key_states, value_states, **cache_kwargs)
        return key, value

    def post_update(self, layer_idx: int, x_len=1, **kwargs):
        self.cache_model[layer_idx].post_update(x_len=x_len, **kwargs)
