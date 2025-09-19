from typing import List, Tuple, Optional
import torch


class HHCache:
    """
    A cache that apply heavy-hitter oracle (https://proceedings.neurips.cc/paper_files/paper/2023/file/6ceefa7b15572587b78ecfcebb2827f8-Paper-Conference.pdf).
    Only the heavy-hitter and the recent tokens are stored in the cache.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the context window.
        num_hh_tokens (`int`):
            The number of heavy hitter tokens. See the original paper for more information.
    """

    def __init__(self, window_length: int, num_hh_tokens: int) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.window_length = window_length
        self.num_hh_tokens = num_hh_tokens
        self.accumulated_attention_scores = None
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx], self.accumulated_attention_scores[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx], self.accumulated_attention_scores[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        accumulated_attention_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens

        if accumulated_attention_scores is not None:
            self.accumulated_attention_scores = (accumulated_attention_scores)

        self._seen_tokens += key_states.shape[-2]
        self.key_cache = torch.cat([self.key_cache, key_states], dim=-2)
        self.value_cache = torch.cat([self.value_cache, value_states], dim=-2)

        return self.key_cache, self.value_cache

    def __call__(
        self,
        attention_scores: torch.Tensor,
        past_key_values,
        num_kv_groups: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Slimming the cache based on accumulated attention scores, only keep heavy-hitters + local tokens.

        Parameters:
            attention_scores (`torch.Tensor`):
                Attention_scores for current steps.
            num_kv_groups (`int`):
                The number of kv groups in repeat kv.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        key_cache, value_cache = past_key_values
        if self.accumulated_attention_scores is None:
            self.accumulated_attention_scores = attention_scores.sum(2)
        else:
            num_new_tokens = attention_scores.shape[2]
            updated_attention_scores = attention_scores.sum(2)# [bs, num_heads, key_len]
            updated_attention_scores[:, :, :-num_new_tokens] += self.accumulated_attention_scores
            self.accumulated_attention_scores = updated_attention_scores

        # Update KV Cache
        if key_cache.shape[-2] > self.window_length:
            seq_scores = self.accumulated_attention_scores[:, :, :-self.window_length + self.num_hh_tokens]
            _, keep_hh_index = torch.topk(seq_scores, self.num_hh_tokens, dim=-1)
            keep_hh_index = keep_hh_index.sort().values

            keep_local_index = torch.arange(key_cache.shape[-2] - self.window_length + self.num_hh_tokens, key_cache.shape[-2], device=keep_hh_index.device).repeat(keep_hh_index.shape[0], keep_hh_index.shape[1], 1)
            keep_index = torch.cat([keep_hh_index, keep_local_index], dim=-1)

            mask = torch.zeros(self.accumulated_attention_scores.shape, dtype=torch.bool).to(keep_hh_index.device)
            mask = mask.scatter(-1, keep_index, 1)

            bsz, num_heads, _, head_dim = key_cache.shape
            mask1 = mask.expand(-1, key_cache.shape[1], -1)
            key_cache = key_cache[mask1].view(bsz, num_heads, -1, head_dim)
            value_cache = value_cache[mask1].view(bsz, num_heads, -1, head_dim)
            self.accumulated_attention_scores = self.accumulated_attention_scores[mask].view(bsz, num_heads, -1)
        return (key_cache, value_cache)


    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx], self.accumulated_attention_scores[layer_idx],))
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, window_length: int, num_hh_tokens: int, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls(window_length, num_hh_tokens)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values) // 3):
                key_states = past_key_values[layer_idx * 3]
                value_states = past_key_values[layer_idx * 3 + 1]
                accumulated_attention_scores = past_key_values[layer_idx * 3 + 2]
                cache.update(key_states, value_states, layer_idx, accumulated_attention_scores=accumulated_attention_scores)
        return cache

    def evict_for_space(self, space_needed: int):
        num_layers = len(self.key_cache)

        # Update score metrics (Accumulated attention scores)
        if len(self.accumulated_attention_scores) < num_layers:
            raise ValueError("The accumulated_attention_scores should be updated before evicting the cache.")

        for layer_idx in range(num_layers):
            # Update KV Cache, Evict for new coming prompts
            if self.get_seq_length(layer_idx) + space_needed > self.window_length:
                if self.window_length - self.num_hh_tokens <= space_needed:
                    raise ValueError("The space_needed should be less than the window_length - num_hh_tokens.")

                seq_scores = self.accumulated_attention_scores[layer_idx][:, :, :-self.window_length + self.num_hh_tokens + space_needed]
                _, keep_hh_index = torch.topk(seq_scores, self.num_hh_tokens, dim=-1)
                keep_hh_index = keep_hh_index.sort().values

                keep_local_index = torch.arange(self.get_seq_length(layer_idx) - self.window_length + self.num_hh_tokens + space_needed, self.get_seq_length(layer_idx), device=keep_hh_index.device).repeat(keep_hh_index.shape[0], keep_hh_index.shape[1], 1)
                keep_index = torch.cat([keep_hh_index, keep_local_index], dim=-1)

                mask = torch.zeros(self.accumulated_attention_scores[layer_idx].shape, dtype=torch.bool).to(keep_hh_index.device)
                mask = mask.scatter(-1, keep_index, 1)

                bsz, num_heads, _, head_dim = self.key_cache[layer_idx].shape
                self.key_cache[layer_idx] = self.key_cache[layer_idx][mask].view(bsz, num_heads, -1, head_dim)
                self.value_cache[layer_idx] = self.value_cache[layer_idx][mask].view(bsz, num_heads, -1, head_dim)
                self.accumulated_attention_scores[layer_idx] = self.accumulated_attention_scores[layer_idx][mask].view(bsz, num_heads, -1)
