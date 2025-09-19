from dataclasses import dataclass
from typing import Callable
import torch


@dataclass
class CacheUpdateInfo:
    b_idx: torch.Tensor | None
    h_idx: torch.Tensor | None

    update_idx: torch.Tensor | None
    shift_idx: torch.Tensor | None

    valid_value_mask: torch.Tensor | None  # This mask is used to remove the unnecessary values
    # b_idx_unique: torch.Tensor
    # h_idx_unique: torch.Tensor
    seq_len: int | None

    size_update_global: torch.Tensor
    size_update_local: torch.Tensor

    global_token_masks: torch.Tensor | None  # This value indicates if the tokens to be preserved are global tokens


def merge_local_to_global_caches(caches: list[torch.Tensor],
                                 updated_cache_info: CacheUpdateInfo,
                                 chunk_size: int = 1,
                                 ):
    """
    This function is applied to put the local KV cache values to the place where global KV cache should stay.
    In the multi head setting, given that different heads might contain different end_seq
    points, some heads might already start a new sequence while the others remain in the same sequence. Once a new
    sequence is provided to the mha layer, we need to check if this new sequence construct a new sub-sequence.
    Then we can compress the new values belonging to the new sequences to the global KV cache values


    Args:
        caches (list[torch.Tensor]): list of torch tensor with size [bsz, n_heads, nctx_l, n_dim]
            This item can be generally considered as two parts: global parts and parts parts. while the first few
            positions store the global KV values. The second parts store the local KV values. If any of the local values
            need to be merged to global tokens, we could extract those required values and put them to the corresponding
            positions

        updated_cache_info (dict[(int, int), torch.Tensor]): a dict that record which local kv cache values will be placed
             in the global KV cache values. the keys are the indices in the batch where the local KV values should
             be updated to the global KV values. The values are the indices in the n_ctx showing which features in
             the concatenated local/current KV values should be compressed into the global KV values.

        chunk_size (int): chunk size, if this value is larger than 1, then we will fold the seq_len dim into
            [-1, chunk_size]
    """
    if updated_cache_info is None or len(updated_cache_info.b_idx) == 0:
        return caches

    b_idx = updated_cache_info.b_idx
    h_idx = updated_cache_info.h_idx

    # before update: we have global + local cache
    # after update: we have only global cache (as local cache is removed)
    # Therefore, the update idx should be cache_size based and fill_idx should be global cache_size based

    update_idx = updated_cache_info.update_idx

    fill_idx = updated_cache_info.shift_idx
    if chunk_size == 1:
        for i, cache in enumerate(caches):
            fill_value = cache[b_idx, h_idx, update_idx].clone()
            cache[b_idx, h_idx, fill_idx] = fill_value
    else:
        for i, cache in enumerate(caches):
            cache = cache.unflatten(2, (-1, chunk_size))
            fill_value = cache[b_idx, h_idx, update_idx].clone()
            cache[b_idx, h_idx, fill_idx] = fill_value
            cache = cache.flatten(2, 3)

    return caches


def reduce_local_cache_with_cache_info(input_cache: torch.Tensor, updated_cache_info: CacheUpdateInfo):
    """
    This function is implemented to get the remaining sliding window tokens after merging the local tokens to global
    tokens
    """
    b_idx_unique = updated_cache_info.b_idx_unique.view(-1, 1)
    h_idx_unique = updated_cache_info.h_idx_unique.view(-1, 1)

    cache_local = input_cache[b_idx_unique, h_idx_unique]
    if updated_cache_info.valid_value_mask is not None:
        cache_local *= updated_cache_info.valid_value_mask
    input_cache[b_idx_unique, h_idx_unique] = cache_local
    return input_cache


def update_valid_tokens(valid_tokens: torch.Tensor,
                        updated_cache_info: CacheUpdateInfo | None,
                        ) -> torch.Tensor:
    """
    Update valid tokens by placing the positions where global tokens should appear to be 1

    """
    if updated_cache_info is None or len(updated_cache_info.b_idx) == 0:
        return valid_tokens

    b_idx = updated_cache_info.b_idx
    h_idx = updated_cache_info.h_idx

    # end_seqs_local only contains local information, so their size should be substracted accordingly
    fill_idx = updated_cache_info.shift_idx

    # if updated_cache_info.global_token_masks is not None:
    #    b_idx = b_idx[updated_cache_info.global_token_masks]
    #    if len(b_idx) == 0:
    #        return valid_tokens
    #    h_idx = h_idx[updated_cache_info.global_token_masks]
    #    fill_idx = fill_idx[updated_cache_info.global_token_masks]
    fill_idx_max = torch.max(fill_idx) + 1

    if fill_idx_max > valid_tokens.shape[2]:
        valid_tokens = torch.nn.functional.pad(
            valid_tokens, (0, (fill_idx_max - valid_tokens.shape[2])), value=-torch.inf
        )
    # global tokens need to be valid
    valid_tokens[b_idx, h_idx, fill_idx] = 0

    return valid_tokens


def restore_global_features(cache: torch.Tensor,
                            bsz: int,
                            n_ctx_global: int):
    """
    This function is used to refill 0 values to the global cache. This aims to restore the modified features from
    construct_features_for_input

    Args:
        cache (torch.tensor): a torch tensor with size [bsz, n_ctx, nh, n_dim] to be cleared
        bsz (int): batch size
        n_ctx_global (int): the number of valid context values in the cache.
    Returns:
        the cleared KV caches
    """
    cache[:, :, n_ctx_global:] = 0.
    return cache


def extend_cache(x: torch.Tensor,
                 cache: torch.Tensor,
                 fill_idx: torch.Tensor,
                 n_ctx_cache: int,
                 dim=2,
                 ):
    """
    We fill the values from the input x to cache_local, cache will be dynamically extended if the items stored within
    cache are smaller than the required n_ctx.
    """
    x = x.to(cache)

    if n_ctx_cache > cache.shape[dim]:
        # TODO replace this with pad!
        new_tensor = torch.zeros(
            [*x.shape[:dim], n_ctx_cache - cache.shape[dim], *x.shape[dim + 1:]], device=cache.device, dtype=cache.dtype
        )
        cache = torch.cat([cache, new_tensor], dim=dim)

    cache.scatter_(dim, fill_idx.to(device=x.device), x)
    return cache


def construct_context_features(x: torch.Tensor,
                               cache: torch.Tensor,
                               size_ctx_local: torch.Tensor,
                               scores_size: int | torch.Tensor,
                               n_ctx_cache: int = 0,
                               ):
    """
    This function is used to construct the feature maps to compute attentions. Here we have 3 feature maps: the
    global cache, the local cache and the new x values. We need to concatenate the 3 tensors to form a new tensor
    that can be fed to the MHA layer. This function is implemented to both assign new values to the local caches and
    construct a context feature. Therefore, we need a temporary tensor to store all this information. Here,
    we will store them into the global cache. and clear all the temporarily added feature maps during the
    postprocessing process.
    Args:
        x (torch.tensor):  torch tensor with size [bsz, nh, n_ctx_new, n_dim]  the new kv values feed to the network.
        cache (torch.tensor): torch tensor with size [bsz, n_heads, nctx_l, n_dim], cached KV values
        n_ctx_global (int): length of valid global features
        n_ctx_local (int): length of valid local features
        size_ctx_local (torch.Tensor): torch tensor with size [bsz] that records the amount of valid values in the
            local caches
        bsz (int): batch size
        x_len (int): length of x
    Returns:
        x_out (torch.tensor): a feature map that is ready to be fed to the attention map
    """
    # we first feed the x to the local cache
    cache = extend_cache(x, cache, size_ctx_local, n_ctx_cache)
    return cache[:, :, :scores_size]
