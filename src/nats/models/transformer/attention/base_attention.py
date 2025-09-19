import math

import torch
import torch.nn.functional as F

from nats.components.cache.dyn_cache import TransformerCache, NAtSCache
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


class AttentionLayer(nn.Module):

    def __init__(self,
                 dim: int = 4096,
                 n_heads: int = 32,
                 n_kv_heads: int | None = None,
                 layer_id: int = 0,
                 dropout: float = 0.0,
                 use_flash: bool = True,
                 on_ddp: bool = False
                 ):
        """
        Base Attention.
        This attention incorporate the application of segment-wise attention.
        This is the same as the vanilla attention except the masks (nats.components.masks) during the training time
        While during test time, we store the following information
             * the Global KV cache. This tensor contained all the Global KV values (where end_seqs equals True)
             * the Local KV cache. This tensor indicates the KV values of the current sequences.
        This approach works well in the single batch setting where all the heads share the same end_seqs singnals:
        during inference, we simply concatenate the cached KV values as the final attention. However, this might not
        work well under the batch setting or mult-head attentions where end_seqs might happen in different places.
        Hence, we propose the following approach:
            * we use masks to control the attentions. Hence, we could fill the placeholder zero tensors to the positions
              where multiple heads disagree with each other and then mask out the corresponding rows in the
              corresponding positions for the corresponding heads. For instance, one could have the following valid
              global KV cache values (where only the tokens that contain at least one end_seq==1 could be stored in this
              global KV cache):
              [[0,1,0,1],
               [1,1,1,0]]
              This indicates that for the first head, the masks should be torch.repeat([[-inf, 0, -inf, 0]], ...).
              This tensor should only store the "pure" global KV values. i.e., none of the tokens stored within the
              global KV cache values are stored in the same sequence as the tensors to be inferenced
              The above approach is designed for single-batch multi head attentions. While for the multi-batch multi
              head attentions, we pad additional zero values to the end of each global KV cache
            * for the local attention KV cache values. This contains the tokens that stored under the same sequence as
              the current tokens
              ** this indicates that, to be stored under the local KV cache values, at least one enq_seq token should
                 contain a sequences that only contain `False`
        To handel the above cases, we need  to store the following information
            * end_seqs_global (tensor of size [bsz, nheads, nctx_global]) that stores if the token in the current head is
              a global token (otherwise, the corresponding masks will be applied)
            * end_seqs_local (tensor of size [bsz, nheads, nctx_local]) that stores if the token in the current head
            * nctx_global (tensor of size [bsz, nheads, 1]) that stores the number valid global KV cache values (given
              that we pad the zero values to the bottom of each global tokens)
            * nctx_local(tensor of size [bsz, nheads, 1] that stores the number of valid local KV cache value.
        During the inference time, once a (or few) new tokens are quired, we first check if a new sub-sequence could be
        provided by the new token: in that case, there will be no sequence that only contains 0 values. We will then
        compress the tokens from the Local KV cache to the Global KV cache. Hence, we
            * first check the token that could provide the longest pure 0 sequence with the new KV values (This ensures
              that we still preserve the longest sub-sequence)
            * push the tokens before the token found in step one to the global KV Caches and refresh the current tokens
              in the Local KV caches


        """
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.layer_id = layer_id

        if on_ddp:
            """
            model_parallel_size = fs_init.get_model_parallel_world_size()
            self.n_local_heads = n_heads // model_parallel_size
            self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
            self.n_rep = self.n_local_heads // self.n_local_kv_heads
            self.head_dim = dim // n_heads
            self.wq = ColumnParallelLinear(
                dim,
                n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            self.wk = ColumnParallelLinear(
                dim,
                self.n_kv_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            self.wv = ColumnParallelLinear(
                dim,
                self.n_kv_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            self.wo = RowParallelLinear(
                n_heads * self.head_dim,
                dim,
                bias=False,
                input_is_parallel=True,
                init_method=lambda x: x,
            )
            """
            # model_parallel_size = fs_init.get_model_parallel_world_size()
            self.n_local_heads = n_heads
            self.n_local_kv_heads = self.n_kv_heads
            self.n_rep = self.n_local_heads // self.n_local_kv_heads
            self.head_dim = dim // n_heads
            self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False, )
            self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False, )
            self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False, )
            self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False, )
        else:
            self.n_local_heads = n_heads
            self.n_local_kv_heads = self.n_kv_heads
            self.n_rep = self.n_local_heads // self.n_local_kv_heads
            self.head_dim = dim // n_heads

            self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False, )
            self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False, )
            self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False, )
            self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False, )
        self.use_flash = use_flash
        self._use_flash = use_flash

        self.dropout = dropout

        self.cache_type = 'vanilla'
        self.attn_dropout = nn.Dropout(dropout)

    def prepare_x_values(self, x: torch.Tensor, bsz: int, seqlen: int,
                         pos_emb: tuple[torch.Tensor, torch.Tensor] | None,
                         start_pos: int = 0, cache: TransformerCache | None = None
                         ):
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # (bs, n_local_heads, seqlen, head_dim)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim).transpose(1, 2)

        xq, xk = self.add_rotary_emb(xq, xk, pos_emb)

        return xq, xk, xv, {}

    def add_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor, pos_emb: tuple[torch.Tensor, torch.Tensor] | None):
        if pos_emb is not None:
            cos, sin = pos_emb
            xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        return xq, xk

    def repeat_kv_values(self, keys: torch.Tensor, values: torch.Tensor):
        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        return keys, values

    def prepare_kv(self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, start_pos: int = 0,
                   cache: NAtSCache | TransformerCache | None = None,
                   **kwargs
                   ):
        if self.training:
            return xk, xv
        else:
            keys, values = cache.update(key_states=xk, value_states=xv, layer_idx=self.layer_id, cache_kwargs=kwargs,)

            return keys, values

    def update_cache(self,
                     bsz: int = 0,
                     x_len: int = 0,
                     cache: NAtSCache | TransformerCache | None = None,
                     **kwargs):
        cache.post_update(layer_idx=self.layer_id, x_len=x_len, **kwargs)

    def multi_head_attention(self, xq: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                             mask: torch.Tensor | None = None,
                             cache: NAtSCache | TransformerCache | None = None,
                             **kwargs,
                             ):
        if self._use_flash and self.training:
            return F.scaled_dot_product_attention(xq, keys, values, attn_mask=mask,
                                                  dropout_p=self.dropout if self.training else 0,
                                                  is_causal=True if mask is None else False)

        scores = torch.matmul(xq, keys.transpose(2, 3)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            mask = mask[:, -scores.shape[-1]:]
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # scores = self.attn_dropout(scores)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)+
        if not self.training:
            cache.post_update(layer_idx=self.layer_id, x_len=xq.shape[-2],
                              attn_weights=scores,
                              output=output)
        return output

    def prepare_cache(self, max_batch_size: int, max_seq_len: int, n_heads: int, head_dims: int, device: torch.device,
                      cache_type: str = 'vanilla',
                      **cache_kwargs,
                      ):
        """
        This function should be called before evaluating
        """
        self.cache_k = torch.zeros(
            (
                max_batch_size,
                n_heads,
                0,
                head_dims,
            ),
            device=device
        )
        self.cache_v = torch.zeros(
            (
                max_batch_size,
                n_heads,
                0,
                head_dims,
            ),
            device=device
        )
        if cache_type != 'vanilla':
            self.cache_type = cache_type
            if cache_type == 'att_sink':
                from nats.models.transformer.attention.baselines.attention_sink import StartRecentKVCache
                start_size = cache_kwargs['start_size']
                recent_size = cache_kwargs['recent_size']
                self.kv_cache = StartRecentKVCache(start_size, recent_size)
            elif cache_type == 'cam':
                from nats.models.transformer.attention.baselines.cam import StartRecentKVCache_cam
                start_size = cache_kwargs['start_size']
                recent_size = cache_kwargs['recent_size']
                self._use_flash = False
                self.kv_cache = StartRecentKVCache_cam(start_size, recent_size)
            elif cache_type == 'h2o':
                from nats.models.transformer.attention.baselines.h2o import HHCache
                num_hh_tokens = cache_kwargs['num_hh_tokens']
                self._use_flash = False
                self.kv_cache = HHCache(2 * num_hh_tokens, num_hh_tokens)
            else:
                raise NotImplementedError(f'Unknown cache type {cache_type}!!!')

    def reset_cache(self):
        self.cache_k.zero_()
        self.cache_v.zero_()

    def del_cache(self):
        del self.cache_k
        del self.cache_v
        self._use_flash = self.use_flash

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: tuple[torch.Tensor, torch.Tensor] | None,
        mask: torch.Tensor | None,
        start_pos: int = 0,
        cache: NAtSCache | TransformerCache | None = None,
        **kwargs,
    ):
        bsz, seqlen, _ = x.shape

        xq, xk, xv, mha_kwargs = self.prepare_x_values(x, bsz=bsz, seqlen=seqlen, pos_emb=pos_emb,
                                                       start_pos=start_pos, cache=cache)

        keys, values = self.prepare_kv(xq, xk, xv, start_pos,
                                       cache=cache,
                                       kwargs=mha_kwargs,)

        keys, values = self.repeat_kv_values(keys, values)
        output = self.multi_head_attention(xq, keys, values, mask=mask, cache=cache,
                                           kwargs=mha_kwargs)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

    def _generate_mask(self, x: torch.Tensor, start_pos: int, **kwargs):
        return None


class GPTAttentionLayer(AttentionLayer):
    """
    https://github.com/karpathy/nanoGPT/blob/master/model.py
    GPT attention is similar to llama attention. However, they do not require additional grouped kv values and
    """

    def __init__(self, dim: int = 4096,
                 n_heads: int = 32,
                 n_kv_heads: int | None = None,
                 layer_id: int = 0,
                 dropout: float = 0.0,
                 use_flash: bool = True,
                 on_ddp: bool = False):
        # we do not need grouped kv cache in GPT model
        # assert n_kv_heads is None
        super(GPTAttentionLayer, self).__init__(dim, n_heads, n_kv_heads, layer_id, dropout, use_flash, on_ddp)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: tuple[torch.Tensor, torch.Tensor] | None,
        mask: torch.Tensor | None,
        start_pos: int = 0,
        cache: NAtSCache | TransformerCache | None = None,
        **kwargs,
    ):
        return self.resid_dropout(
            super(GPTAttentionLayer, self).forward(x, pos_emb, mask, start_pos=start_pos, cache=cache,
                                                   **kwargs))

    def repeat_kv_values(self, keys: torch.Tensor, values: torch.Tensor):
        return keys, values


class LLAMA3AttentionLayer(AttentionLayer):
    def __init__(self, dim: int = 4096,
                 n_heads: int = 32,
                 n_kv_heads: int | None = None,
                 layer_id: int = 0,
                 dropout: float = 0.0,
                 use_flash: bool = True,
                 on_ddp: bool = False):
        # we do not need grouped kv cache in GPT model
        assert dropout == 0.
        super(LLAMA3AttentionLayer, self).__init__(dim, n_heads, n_kv_heads, layer_id, dropout, use_flash, on_ddp)
