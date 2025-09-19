from typing import Optional

import torch

from torch import nn

from nats.models.model_configuration import TransformerArgs
from nats.models.transformer.attention.base_attention import LLAMA3AttentionLayer, GPTAttentionLayer, AttentionLayer
from nats.models.transformer.attention.nats_attention import LlamaNAtSAttentionLayer, GPTNAtSAttentionLayer
from nats.models.transformer.attention.nats_chunk_attention import (
    LlamaNAtSChunkAttentionLayer,
    GPTNAtSChunkAttentionLayer
)
from nats.components.cache.dyn_cache import TransformerCache, NAtSCache
from nats.models.transformer.components import GPTFeedForward, LLAMA3FeedForward, RMSNorm, LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: TransformerArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = self.get_attention_layer(args, layer_id)
        self.feed_forward = self.get_forward_net(args)
        self.attention_norm = self.get_norm_layer(args)
        self.ffn_norm = self.get_norm_layer(args)

        self.layer_id = layer_id

    @staticmethod
    def get_attention_layer(args: TransformerArgs, layer_id: int = 0) -> AttentionLayer:
        raise NotImplemented

    @staticmethod
    def get_forward_net(args: TransformerArgs):
        raise NotImplemented

    @staticmethod
    def get_norm_layer(args: TransformerArgs):
        raise NotImplemented

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: tuple[torch.Tensor, torch.Tensor],
        mask: Optional[torch.Tensor],
        start_pos: int,
        cache: TransformerCache | NAtSCache | None = None,
        is_first_module: bool = True,
        on_vanilla_attention: bool = False,
        **kwargs
    ):
        h = x + self.attention(self.attention_norm(x), pos_emb, mask, start_pos=start_pos,
                               cache=cache,
                               is_first_module=is_first_module,
                               on_vanilla_attention=on_vanilla_attention,
                               **kwargs)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def generate_cache(self, max_batch_size: int, max_seq_len, device: torch.device, **kwargs):
        self.attention.prepare_cache(max_batch_size, max_seq_len, self.n_heads, self.head_dim, device=device, **kwargs)

    def reset_cache(self):
        self.attention.reset_cache()

    def del_cache(self):
        self.attention.del_cache()


class LLAMA3Block(TransformerBlock):
    @staticmethod
    def get_attention_layer(args: TransformerArgs, layer_id: int = 0):
        if args.nats_enable:
            if args.chunk_size == 1:
                return LlamaNAtSAttentionLayer(
                    layer_id=layer_id,
                    dim=args.dim,
                    n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    use_flash=True,
                    on_ddp=args.on_ddp,
                    n_msks=args.n_msks,
                    sparse_regularized_value=args.sparse_regularized_value,
                    local_seq_max_length=args.local_seq_max_length,
                )
            else:
                return LlamaNAtSChunkAttentionLayer(
                    layer_id=layer_id,
                    dim=args.dim,
                    n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    use_flash=True,
                    on_ddp=args.on_ddp,
                    n_msks=args.n_msks,
                    sparse_regularized_value=args.sparse_regularized_value,
                    local_seq_max_length=args.local_seq_max_length,
                    chunk_size=args.chunk_size,
                    chunk_merge_method=args.chunk_merge_method,
                    compress_on_q=args.compress_on_q,
                )
        return LLAMA3AttentionLayer(
            layer_id=layer_id,
            dim=args.dim,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            use_flash=args.use_flash,
            on_ddp=args.on_ddp,
        )

    @staticmethod
    def get_forward_net(args: TransformerArgs):
        return LLAMA3FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            ffn_hidden_dim=args.ffn_hidden_dims,
            on_ddp=args.on_ddp,
        )

    @staticmethod
    def get_norm_layer(args: TransformerArgs):
        return RMSNorm(args.dim, eps=args.norm_eps)


class GPTBlock(TransformerBlock):
    @staticmethod
    def get_attention_layer(args: TransformerArgs, layer_id: int = 0):
        if args.nats_enable:
            if args.chunk_size == 1:
                return GPTNAtSAttentionLayer(
                    layer_id=layer_id,
                    dim=args.dim,
                    n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    use_flash=True,
                    on_ddp=args.on_ddp,
                    n_msks=args.n_msks,
                    sparse_regularized_value=args.sparse_regularized_value,
                    local_seq_max_length=args.local_seq_max_length,
                )
            else:
                return GPTNAtSChunkAttentionLayer(
                    layer_id=layer_id,
                    dim=args.dim,
                    n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    use_flash=True,
                    on_ddp=args.on_ddp,
                    n_msks=args.n_msks,
                    sparse_regularized_value=args.sparse_regularized_value,
                    local_seq_max_length=args.local_seq_max_length,
                    chunk_size=args.chunk_size,
                    chunk_merge_method=args.chunk_merge_method,
                    compress_on_q=args.compress_on_q,
                )
        return GPTAttentionLayer(
            layer_id=layer_id,
            dim=args.dim,
            n_heads=args.n_heads,
            n_kv_heads=None,
            dropout=args.dropout,
            on_ddp=args.on_ddp
        )

    @staticmethod
    def get_forward_net(args: TransformerArgs):
        return GPTFeedForward(
            dim=args.dim,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            bias=args.ffn_bias,
            on_ddp=args.on_ddp,
            dropout=args.dropout
        )

    @staticmethod
    def get_norm_layer(args: TransformerArgs):
        return LayerNorm(args.dim, args.ln_bias)
