import torch

import math

import wandb
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from nats.models.model_configuration import TransformerArgs, CacheArgs
from nats.components.masks.utils import gumbel_sigmoid
from nats.components.masks.utils import generate_hard_seq_masks, generate_soft_seq_masks
from nats.components.cache.dyn_cache import TransformerCache, NAtSCache

from nats.components.cache.cache_update import update_valid_tokens, CacheUpdateInfo, extend_cache
from nats.models.transformer.blocks import LLAMA3Block, GPTBlock
from nats.models.transformer.components import RMSNorm, LayerNorm
from nats.models.transformer.utils import repeat_masks


class Transformer(nn.Module):

    def __init__(self, params: TransformerArgs, cache_args: CacheArgs):
        super().__init__()
        self.n_layers = params.n_layers

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(self.get_transformer_block(layer_id, params))

        if params.has_output_norm:
            self.norm = self.get_norm_layer(params)
        else:
            self.norm = nn.Identity()

        self.apply_pos_emb = params.apply_pos_emb
        self.cache_args = cache_args

        self.nats_enable = params.nats_enable
        if self.nats_enable:
            self.n_msks = params.n_msks or params.n_kv_heads or params.n_heads
            self.n_rep_msk = params.n_heads // self.n_msks

        self._device = torch.device('cpu')

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self.to(device)
        self._device = device

    def get_transformer_block(self, layer_id: int, params: TransformerArgs):
        raise NotImplemented

    def get_norm_layer(self, params: TransformerArgs):
        raise NotImplemented

    @staticmethod
    def generate_casual_mask(seqlen: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device)

        mask = torch.triu(mask, diagonal=1)
        return mask

    def generate_mask(self, x: torch.Tensor, start_pos: int, **kwargs):
        """
        This function is used to generate the masks
        """
        bz, x_len = x.shape[:2]
        mask = None
        if x_len > 1 or start_pos > 0:
            mask = torch.full((x_len, x_len), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1)
            if self.training:
                return mask
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((x_len, start_pos), device=x.device), mask]
            ).type_as(x)
        return mask

    def prepare_pos_encoding(self, x: torch.Tensor, start_pos: int):
        _bsz, seqlen, _ = x.shape
        # h = self.tok_embeddings(x)
        if self.freqs_cis is not None:
            self.freqs_cis = self.freqs_cis.to(x.device)
            freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]
        else:
            freqs_cis = None

        return x, freqs_cis

    def forward(self,
                x: torch.Tensor,
                start_pos: int = 0,
                cache: TransformerCache | NAtSCache | None = None,
                on_vanilla_attention: bool = False,
                **kwargs):
        h, freqs_cis = self.prepare_pos_encoding(x, start_pos)

        mask = None
        if not self.nats_enable or on_vanilla_attention:
            mask = self.generate_mask(h, start_pos=start_pos, **kwargs)

        for i, layer in enumerate(self.layers):
            h = layer(h, freqs_cis, mask, start_pos=start_pos, cache=cache,
                      )

        if self.nats_enable and self.training and not kwargs['on_val']:
            fraction_size = {
                f'msk/fraction_{layer.attention.layer_id}': layer.attention.end_seqs_size.detach() for layer in self.layers
            }
            wandb.log(fraction_size)

        h = self.norm(h)
        return h

    def generate_cache(self, max_batch_size: int, max_seq_len: int, device: torch.device, **kwargs):
        for layer in self.layers:
            layer.generate_cache(max_batch_size, max_seq_len, device=device, **kwargs)

    def reset_cache(self):
        for layer in self.layers:
            layer.reset_cache()

    def del_cache(self):
        for layer in self.layers:
            layer.del_cache()


class LLAMA3Transformer(Transformer):
    def __init__(self, params: TransformerArgs, cache_args: CacheArgs):
        super(LLAMA3Transformer, self).__init__(params, cache_args)
        if self.apply_pos_emb and params.rope_type is not None:
            self.rotary_emb = LlamaRotaryEmbedding(config=params)
        else:
            # self.freqs_cis = None
            self.rotary_emb = None

    def get_transformer_block(self, layer_id: int, params: TransformerArgs):
        return LLAMA3Block(layer_id, params)

    def get_norm_layer(self, params: TransformerArgs):
        return RMSNorm(params.dim, eps=params.norm_eps)

    def prepare_pos_encoding(self, x: torch.Tensor, start_pos: int):
        pos_emb = None
        if self.apply_pos_emb:
            _bsz, seqlen, _ = x.shape
            pos = torch.arange(start_pos, start_pos + seqlen, device=x.device).unsqueeze(0)
            pos_emb = self.rotary_emb(x, pos)  # position embeddings of shape (t, n_embd)
        return x, pos_emb


class GPTTransformer(Transformer):
    def __init__(self, params: TransformerArgs, cache_args: CacheArgs):
        super(GPTTransformer, self).__init__(params, cache_args)
        # positional encoding layer
        if self.apply_pos_emb:
            self.rotary_emb = LlamaRotaryEmbedding(config=params)
        else:
            # self.freqs_cis = None
            self.rotary_emb = None

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_transformer_block(self, layer_id: int, params: TransformerArgs):
        return GPTBlock(layer_id, params)

    def get_norm_layer(self, params: TransformerArgs):
        return LayerNorm(params.dim, )

    # def prepare_pos_encoding(self, x: torch.Tensor, start_pos: int):
    #     if self.apply_pos_emb:
    #         _bsz, seqlen, _ = x.shape
    #         pos = torch.arange(start_pos, start_pos + seqlen, device=x.device)
    #         pos_emb = self.wpe(pos)  # position embeddings of shape (t, n_embd)
    #         x = self.dropout(x + pos_emb)
    #     return x, None
    def prepare_pos_encoding(self, x: torch.Tensor, start_pos: int):
        pos_emb = None
        if self.apply_pos_emb:
            _bsz, seqlen, _ = x.shape
            pos = torch.arange(start_pos, start_pos + seqlen, device=x.device).unsqueeze(0)
            pos_emb = self.rotary_emb(x, pos)  # position embeddings of shape (t, n_embd)
        return x, pos_emb
