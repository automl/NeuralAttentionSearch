from pathlib import Path

from transformers.cache_utils import DynamicCache
import numpy as np
import torch
from torch import nn

from nats.components.cache.dyn_cache import (
    TransformerCache,
    NAtSCache
)
from nats.models.transformer.transformer import (
    GPTTransformer,
    LLAMA3Transformer,
    Transformer
)
from nats.models.model_configuration import TransformerArgs, CacheArgs
from nats.utils import check_fp16_dtype


class BaseArchitecture(nn.Module):
    def __init__(self, tokenizer_vocab_size: int, model_dim: int, share_embeddings: bool = True):
        super(BaseArchitecture, self).__init__()
        self.tok_embeddings = nn.Embedding(
            tokenizer_vocab_size, model_dim,
        )
        self.lm_head = nn.Linear(
            model_dim, tokenizer_vocab_size, bias=False
        )
        self.fp16_type = torch.float16 if check_fp16_dtype() == 'float16' else torch.bfloat16

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless.
        if share_embeddings:
            self.tok_embeddings.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying
        self._device = torch.device('cpu')
        self.apply(self._init_embeddings)

    def get_backbone_modules(self):
        raise NotImplemented

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self.to(device)
        self._device = device
        backbones = self.get_backbone_modules()
        for backbone in backbones:
            backbone.device = device

    def generate_cache(self, *args, **kwargs):
        raise NotImplemented

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, output_logit: bool = True, on_val: bool = False,
                cache: DynamicCache | None = None,
                on_vanilla_attention: bool = False,
                **kwargs):
        embeddings = self.tok_embeddings(tokens)
        backbone_out = self.backbone_forward(embeddings, start_pos, on_val, cache=cache,
                                             on_vanilla_attention=on_vanilla_attention,
                                             **kwargs)
        if not output_logit:
            return backbone_out
        logit = self.get_logit(backbone_out, on_val=on_val, )
        return logit

    def get_logit(self, transformer_out: torch.Tensor, on_val: bool = False):
        return self.lm_head(transformer_out)

    def backbone_forward(self, embeddings: torch.Tensor, start_pos: int, on_val: bool = False,
                         cache: DynamicCache | None = None,
                         on_vanilla_attention: bool = False,
                         **kwargs):
        raise NotImplemented

    def _init_embeddings(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class TransformerArchitecture(BaseArchitecture):
    def __init__(self, tokenizer_vocab_size: int, transformer_args: TransformerArgs, cache_args: CacheArgs):
        if transformer_args.model_type == 'llama3':
            share_embeddings = False
        else:
            share_embeddings = True
        super(TransformerArchitecture, self).__init__(tokenizer_vocab_size, transformer_args.dim,
                                                      share_embeddings=share_embeddings)
        self.transformer = self.construct_transformer(transformer_args, cache_args)
        self.transformer_args = transformer_args
        self.cache_args = cache_args

    def get_backbone_modules(self):
        return [self.transformer]

    def backbone_forward(self, embeddings: torch.Tensor, start_pos: int, on_val: bool = False,
                         cache: DynamicCache | None = None,
                         on_vanilla_attention: bool = False,
                         **kwargs):
        # for triton based transformers
        if self.device == torch.device('cpu'):
            return self.transformer(embeddings, start_pos=start_pos, on_val=on_val, cache=cache,
                                    on_vanilla_attention=on_vanilla_attention,
                                    **kwargs)
        res = self.transformer(embeddings.to(self.fp16_type), start_pos=start_pos, on_val=on_val, cache=cache,
                               on_vanilla_attention=on_vanilla_attention,
                               **kwargs)
        return res

    @torch.no_grad()
    def backbone_vanilla_transformer_forward(self, embeddings: torch.Tensor, start_pos: int, on_val: bool = False,
                                             cache: DynamicCache | None = None, **kwargs):
        res = self.transformer.forward_vanilla_transformer(embeddings.to(self.fp16_type),
                                                           start_pos=start_pos, on_val=on_val, cache=cache,
                                                           **kwargs)
        return res

    def construct_transformer(self, transformer_args: TransformerArgs, cache_args: CacheArgs,
                              is_first_module: bool = True) -> Transformer:
        if transformer_args.model_type == 'gpt':
            return GPTTransformer(transformer_args, cache_args, )
        elif transformer_args.model_type == 'llama3':
            return LLAMA3Transformer(transformer_args, cache_args, )
        else:
            raise NotImplementedError

    def crop_block_size(self, block_size: int):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def generate_cache(self, cache_type: str = 'transformer', cache_init_kwargs={}, **kwargs):
        if self.transformer_args.nats_enable:
            return NAtSCache(self.transformer_args.local_seq_max_length, self.transformer_args.chunk_size)
        else:
            return TransformerCache(cache_type=cache_type,
                                    n_kv_heads=self.transformer_args.n_kv_heads or self.transformer_args.n_heads,
                                    cache_init_kwargs=cache_init_kwargs,
                                    )
