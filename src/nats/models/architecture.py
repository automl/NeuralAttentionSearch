import os
import json

from typing import Callable
from dataclasses import asdict
from pathlib import Path

from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin

import torch
from torch import nn
from torch.nn import functional as F

from nats.models.transformer.components import NormLayerAdapter, LayerNorm, RMSNorm
from nats.training.optimizer import OptimizerArgs, get_optimizer
from nats.models.model_configuration import TransformerArgs, CacheArgs
from nats.models.backbone import TransformerArchitecture


class LanguageModels(nn.Module, GenerationMixin):
    """
    This function is mainly adapted from the nanogpt:
    https://github.com/karpathy/nanoGPT/blob/master/model.py
    """

    def __init__(self, tokenizer_vocab_size: int, transformer_args:TransformerArgs,
                 cache_args: CacheArgs,
                 ):
        super(LanguageModels, self).__init__()
        self.transformer_needs_sparsity = False
        self.model = TransformerArchitecture(tokenizer_vocab_size, transformer_args, cache_args)

        self.has_cache = False
        # self.transformer_cache_max_bz = transformer_args.max_batch_size
        # self.transformer_cache_max_seql = transformer_args.max_seq_len

        self.meta_info = {
            'tokenizer_vocab_size': tokenizer_vocab_size,
            'transformer_args': asdict(transformer_args),
        }

        self._device = torch.device('cpu')

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self.to(device)
        self.model.device = device

    def save(self, base_path: Path):
        if not base_path.exists():
            os.makedirs(base_path)
        with open(base_path / 'meta_info.json', 'w') as f:
            json.dump(self.meta_info, f)
        torch.save(self.state_dict(), base_path / 'model_weights.pth')

    @staticmethod
    def load(base_path: Path, device=torch.device('cuda')) -> 'LanguageModels':
        with open(base_path / 'meta_info.json') as f:
            meta_info = json.load(f)

        tokenizer_vocab_size = meta_info['tokenizer_vocab_size']
        model = LanguageModels(tokenizer_vocab_size, transformer_args=TransformerArgs(**meta_info['transformer_args']),
                               cache_args=CacheArgs(n_kv_heads=meta_info['transformer_args']['n_heads'],
                               n_msks=meta_info['transformer_args']['n_heads'],
                               ))
        model.device = device

        model.load_state_dict(torch.load(base_path / 'model_weights.pth', map_location=device), )
        return model

    def get_num_params(self, non_embedding: bool = True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, start_pos: int = 0, n_outputs: int | None = None,
                return_logit: bool = True,
                on_val: bool = False,
                cache: DynamicCache | None = None,
                on_vanilla_attention: bool = False,
                **model_kwargs):
        if targets is not None:
            logits = self.model(idx, start_pos=start_pos, on_val=on_val, cache=cache,
                                on_vanilla_attention=on_vanilla_attention,
                                **model_kwargs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            if self.transformer_needs_sparsity:
                end_seq_size = torch.mean(self.model.get_end_seqs_values())
                loss = end_seq_size * self.regularize_end_seq
            return logits, loss
        else:
            model_out = self.model(idx, start_pos=start_pos, output_logit=False, on_val=on_val,
                                   cache=cache,
                                   on_vanilla_attention=on_vanilla_attention,
                                   **model_kwargs)
            if return_logit:
                if n_outputs is None:
                    logits = self.model.get_logit(model_out)
                else:
                    logits = self.model.get_logit(model_out[:, -n_outputs:])
            else:
                logits = model_out
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # this is done under
            loss = None
            return logits, loss, model_out

    @torch.no_grad()
    def grad_norm(self):
        total_norm = 0.0
        for n, p in self.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def generate_cache(self, bsz, max_seq_len, **kwargs):
        return self.model.generate_cache(bsz=bsz, max_seq_len=max_seq_len, **kwargs)

    def configure_optim_groups(self, weight_decay) -> list[dict]:
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        return self.configurate_optim_groups_for_parameters(param_dict, weight_decay)

    @staticmethod
    def check_if_par_requrie_wd(par_name: str, par: nn.Parameter) -> bool:
        return par.dim() >= 2

    @staticmethod
    def configurate_optim_groups_for_parameters(param_dict: dict, weight_decay: float,
                                                filter_func: Callable | None = None):
        if filter_func is None:
            filter_func = LanguageModels.check_if_par_requrie_wd
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if filter_func(n, p)]
        nodecay_params = [p for n, p in param_dict.items() if not filter_func(n, p)]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        return optim_groups

    def configurate_optimizer(self, optimizer_args: OptimizerArgs):
        optim_group = self.configure_optim_groups(optimizer_args.weight_decay)
        optimizer = get_optimizer(optimizer_args, optim_group)
        return optimizer