import torch


class TransformerLayerCache:
    def __init__(self, **kwargs):
        self.cache_k = None
        self.cache_v = None

    def update(self,
               key_states: torch.Tensor,
               value_states: torch.Tensor,
               **kwargs,
               ):
        if self.cache_k is None:
            self.cache_k = key_states
            self.cache_v = value_states
        else:
            self.cache_k = torch.cat([self.cache_k, key_states], dim=-2)
            self.cache_v = torch.cat([self.cache_v, value_states], dim=-2)
        return self.cache_k, self.cache_v

    def post_update(self, x_len: int, **kwargs):
        return

    def get_seq_length(self):
        if self.cache_k is None:
            return 0
        return self.cache_k.shape[-2]
