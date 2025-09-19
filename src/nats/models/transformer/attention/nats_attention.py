import math

import torch
from torch import nn
import torch.nn.functional as F

from nats.models.transformer.attention.base_attention import GPTAttentionLayer, LLAMA3AttentionLayer, AttentionLayer
from nats.components.cache.dyn_cache import TransformerCache, NAtSCache
from nats.models.transformer.utils import repeat_masks

from nats.models.transformer.triton.flashattention_mask_on_chip import nats_attention


class BaseNAtSAttentionLayer(AttentionLayer):
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
                 global_token_enhanced_value: float = 0.1
                 ):
        super(BaseNAtSAttentionLayer, self).__init__(dim, n_heads, n_kv_heads, layer_id, dropout, use_flash, on_ddp)
        if n_msks is None:
            n_msks = n_kv_heads or n_heads
        self.proj_layer = nn.Linear(dim, n_msks * n_options, bias=False)
        # self.proj_layer = nn.Sequential(
        #   nn.Linear(self.head_dim, 4 * self.head_dim, bias=False),
        #   nn.SELU(),
        #   nn.Linear(4 * self.head_dim, 1, bias=False)
        # )
        self.n_msks = n_msks
        self.n_options = n_options
        self.n_rep_msk = n_heads // n_msks
        self.sm_scale = 1 / math.sqrt(self.head_dim)

        self.sparse_regularized_value = sparse_regularized_value
        self.local_seq_max_length = local_seq_max_length

        # self.proj_layer_norm = LayerNorm(dim)
        # Due to the precision issue, the logits computed by the cumulative mean x value might not be correct and
        # and sometimes we might mis-classify some tokens. To avoid that, we add an additional
        # global_token_enhanced_value to enhance the logits for glboal tokens
        #
        self.global_token_enhanced_value = global_token_enhanced_value

        self.end_seqs_size = torch.tensor([0])
        one_hot_values = torch.zeros(n_options, )
        one_hot_values[0] = 1.
        self.one_hot_values = one_hot_values
        self._cache_is_till_updating = None
        self.executor = None

    def _sample_for_token_info_values(self, x: torch.Tensor, ):
        # """
        logit = self.proj_layer(x.detach()).unflatten(-1, (self.n_msks, self.n_options))
        samples = nn.functional.gumbel_softmax(logit, tau=1., hard=True, dim=-1)  # (B, N_CTX, H, N_OPTs)

        # the first and the last elements are assigned to the first element
        self.one_hot_values = self.one_hot_values.to(samples)
        samples[:, [0, -1], :] = self.one_hot_values
        samples = torch.transpose(samples, 1, 2).contiguous()  # (B, H, N_CTX, N_OPTS)

        return samples

    def prepare_x_values(self, x: torch.Tensor, bsz: int, seqlen: int,
                         pos_emb: tuple[torch.Tensor, torch.Tensor] | None,
                         start_pos: int = 0, cache: NAtSCache | None = None):
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # (bs, n_local_heads, seqlen, head_dim)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim).transpose(1, 2)

        if self.training:
            token_states_info = self._sample_for_token_info_values(x)
        else:
            token_states_info = self._sample_for_token_info_values_inference(x, start_pos=start_pos, cache=cache)

        xq, xk = self.add_rotary_emb(xq, xk, pos_emb)

        return xq, xk, xv, {'token_states_info': token_states_info}

    def _sample_for_token_info_values_inference(self, x: torch.Tensor, start_pos, cache: NAtSCache | None = None,
                                                sink_token_end: int = 1):
        logit = self.proj_layer(x).unflatten(-1, (self.n_msks, self.n_options))
        logit[..., 0] += self.global_token_enhanced_value

        # in this case, there is no need to do gumble softmax
        index = logit.max(-1, keepdim=True)[1]
        if start_pos < sink_token_end:
            index[:, 0] = 0

        samples = torch.zeros_like(logit, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        samples = torch.transpose(samples, 1, 2).contiguous()  # (B, H, N_CTX, N_OPTS)

        return samples

    def get_end_seq_indices(self, end_seqs: torch.Tensor):
        seq_len = end_seqs.shape[2]
        return seq_len - 1 - torch.flip(torch.cummax(torch.flip(end_seqs, (-1,)), -1)[1], (-1,))

    def prepare_kv(self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, start_pos: int = 0,
                   cache: NAtSCache | None = None,
                   kwargs: dict = None
                   ):
        if self.training:
            return xk, xv
        else:
            if self._cache_is_till_updating is not None:
                self._cache_is_till_updating.result()

            keys, values = cache.update(key_states=xk, value_states=xv, layer_idx=self.layer_id,
                                        cache_kwargs=kwargs)

            return keys, values

    def repeat_kv_values(self, keys: torch.Tensor, values: torch.Tensor):
        if self.training:
            return keys, values
        else:
            return super().repeat_kv_values(keys, values)

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
            n_valid_tokens = torch.where(token_states_info[..., -1] == 1., self.local_seq_max_length, n_valid_tokens)

            self.end_seqs_size = torch.mean(n_valid_tokens / N_CTX * 2)

            res = nats_attention(
                xq.contiguous(), keys.contiguous(), values.contiguous(), token_states_info, end_seq_idx,
                causal=True, sm_scale=self.sm_scale,
                n_rep=self.n_rep,
                sparse_regularized_value=self.sparse_regularized_value,
                local_seq_max_length=self.local_seq_max_length,
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
                    cache.post_update, layer_idx=self.layer_id, x_len=x_len, is_first_module=is_first_module,
                )
            else:
                cache.post_update(layer_idx=self.layer_id, x_len=x_len, is_first_module=is_first_module)
            return output


class GPTNAtSAttentionLayer(BaseNAtSAttentionLayer, GPTAttentionLayer):
    pass


class LlamaNAtSAttentionLayer(BaseNAtSAttentionLayer, LLAMA3AttentionLayer):
    pass
