# the main part of this function is adapted from duoattention with MIT license:
# https://github.com/mit-han-lab/duo-attention/blob/main/duo_attn/train.py
import json
import dataclasses
import os
from pathlib import Path
from typing import Any
import numpy as np


import torch
import math

import torchtune.utils
import wandb
import datasets
from dataclasses import asdict, dataclass
from dataclasses import dataclass
from torch import nn
from tqdm import tqdm

import torch.distributed as dist
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

from nats.utils import get_kwargs
from nats.training.multi_gpus import multi_gpu_setup
from nats.training.optimizer import OptimizerArgs
from nats.models.architecture import LanguageModels
from nats.models.transformer.blocks import TransformerBlock

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from nats.utils import check_fp16_dtype
from nats.training.multi_gpus import apply_fsdp
from torchtune import training
from torchtune.modules.peft import get_adapter_state_dict

# FUll fine tune !!!
from torch.distributed.checkpoint.state_dict import (
    _init_optim_state,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torchao.dtypes.nf4tensor import NF4Tensor, to_nf4
from transformers import PreTrainedModel


def _gather_nf4_tensor(sharded_param: nn.Parameter) -> nn.Parameter:
    """
    Manually gather NF4Tensor parameter since it does not support all_gather
    """
    mesh = sharded_param.device_mesh
    nf4_tensor = sharded_param._local_tensor
    quant_params, metadata = nf4_tensor.fsdp_pre_all_gather(mesh)
    full_quant_params = []
    for quant_param in quant_params:
        d0, *dn = quant_param.shape
        shape = (d0 * mesh.get_group().size(), *dn)
        full_quant_param = torch.empty(
            shape, device=quant_param.device, dtype=quant_param.dtype
        )
        dist.all_gather_into_tensor(
            full_quant_param, quant_param, mesh.get_group(), async_op=False
        )
        full_quant_params.append(full_quant_param)
    full_param, _ = nf4_tensor.fsdp_post_all_gather(
        full_quant_params, metadata, nf4_tensor.dtype
    )
    return full_param


def gather_cpu_state_dict(
    model: "FSDPModule",  # noqa
    is_rank_zero: bool,
    device: torch.device | None = None,
    adapter_weights_only: bool = False,
) -> dict[str, Any]:
    """
    Converting sharded state dict into a full state dict on CPU
    Returning non-empty result only on rank0 to avoid peaking CPU memory
    Currenltly we can used distributed state dict API to process model without NF4Tensor. Otherwise, we need to
    manually gather any NF4 tensors until all-gather is supported in the NF4Tensor subclass
    TODO: add support for NF4Tensor at distributed state dict API

    Args:
        model (FSDPModule): Model to generate fully qualified names for cpu_state_dict
        is_rank_zero (bool): flag to check if the process is on rank 0
        device (Optional[torch.device]): device to use for sharded tensors. Default: None
        adapter_weights_only (bool): flag to check if only trainable parameters should be returned. Default: False

    Returns:
        Dict[str, Any]: State dict on CPU
    """
    cpu_state_dict = {}
    sharded_sd = model.state_dict()
    has_nf4 = any(
        hasattr(param, "_local_tensor") and isinstance(param._local_tensor, NF4Tensor)
        for param in sharded_sd.values()
    )
    if has_nf4:
        cpu_state_dict = {}
        sharded_sd = model.state_dict()
        for param_name, param in sharded_sd.items():
            if param.is_cpu:
                # Move back to device if offloaded to CPU
                param = param.to(device)
            if hasattr(param, "_local_tensor"):
                if isinstance(param._local_tensor, NF4Tensor):
                    param = _gather_nf4_tensor(param)
                else:
                    # Gather DTensor
                    param = param.full_tensor()
            if isinstance(param, NF4Tensor):
                param = param.to(param.dtype)
            if is_rank_zero:
                cpu_state_dict[param_name] = param.cpu()
            torch.distributed.barrier()
        return cpu_state_dict
    else:
        options = StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            cpu_offload=True,
        )
        cpu_state_dict = get_model_state_dict(model=model, options=options)
        if adapter_weights_only:
            adapter_key_filter = lambda x: "lora" in x or "magnitude" in x or 'tuning_layer' in x or 'proj_layer' in x
            cpu_state_dict = {k: v.to(device) for k, v in cpu_state_dict.items() if adapter_key_filter(k)}
        if is_rank_zero:
            return cpu_state_dict
        else:
            return {}


class HFLongBenchDistillTrainer:
    def __init__(self,
                 model: PreTrainedModel,
                 optimizer: torch.optim.Optimizer,
                 optimizer_args: OptimizerArgs,
                 tokenizer: AutoTokenizer,
                 dataset_path: Path,
                 model_path: Path,
                 data_loader: torch.utils.data.DataLoader,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                 lr_scheduler_type: str = 'LambdaLR',
                 gradient_accumulation_steps: int = 1,
                 model_lm_head_weight : torch.Tensor | None = None,
                 batch_size: int = 64,
                 grad_clip: float = 1.0,
                 warmup_iters: int = 2000,
                 lr_decay_iters: int = 600000,  # should be ~= max_iters per Chinchilla
                 max_iters: int = 600000,
                 eval_iters: int = 200,
                 eval_interval: int = 2000,
                 lr_max: float = 6e-4,  # max learning rate
                 lr_min: float = 6e-5,  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
                 use_lora: bool = False,
                 use_multi_gpus: bool = False,
                 rank: int = 0,
                 local_rank: int = 0,
                 world_size: int = 1,
                 iter_number: int = 0,
                 log_wandb: bool = True,
                 eval_only: bool = False,
                 on_cpu: bool = False,
                 save_after_eval: bool = True,
                 train_on_labels:bool=True,
                 best_val_loss: float = 1e9,
                 ):
        all_kwargs = get_kwargs()
        self.optimizer = optimizer

        self.model = model

        self.batch_size = batch_size

        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.warmup_iters = warmup_iters
        self.eval_iters = eval_iters
        self.eval_interval = eval_interval
        self.max_iters = max_iters
        self.lr_decay_iters = lr_decay_iters

        self.lr_min = lr_min
        self.lr_max = lr_max

        if on_cpu:
            self.device = torch.device('cpu')
        else:
            #torchtune.utils.get_device()
            self.device = torch.device(type="cuda", index=torch.cuda.current_device())
        #self.model.device = self.device

        self.dataset_path = Path(dataset_path)

        self.dataset_path = Path(dataset_path)

        self.model_path = Path(model_path)
        self.opt_model_path = Path(str(model_path) + '_opt')
        self.use_lora = use_lora

        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size

        self.tokenizer = tokenizer
        self.train_loader = data_loader

        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_type = lr_scheduler_type
        self.model_lm_head_weight = model_lm_head_weight

        self.use_multi_gpus = use_multi_gpus
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
        device_name = torch.cuda.get_device_name()
        unsupported_gpus = ['RTX 20', 'RTX 10']
        for unsupported_gpu in unsupported_gpus:
            if unsupported_gpu in device_name:
                dtype = 'float16'

        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

        self.log_wandb = log_wandb
        self.eval_only = eval_only

        self.grad_clip = grad_clip
        self.iter_num = iter_number
        self.best_val_loss = best_val_loss

        self.save_after_eval = save_after_eval

        self.meta_info = {
            key: str(value) if isinstance(value, Path) else value for key, value in all_kwargs.items() if
            (isinstance(value, (int, list, float, str, Path)) or value is None)
        }
        if dataclasses.is_dataclass(optimizer_args):
            opt_args = dataclasses.asdict(optimizer_args)
        elif isinstance(dataclasses, dict):
            opt_args = optimizer_args
        else:
            raise NotImplemented(f'Unknown optimizer args type: {type(optimizer_args)}')

        self.meta_info['optimizer_args'] = opt_args
        self.train_on_labels = train_on_labels

    def save(self, base_path: Path | str, adapter_params: dict | None = None, optimizer_params: dict | None = None):
        if not isinstance(base_path, Path):
            base_path = Path(base_path)
        if not base_path.exists():
            os.makedirs(base_path, exist_ok=True)
        model_path = base_path / 'model'
        if not model_path.exists():
            os.makedirs(model_path, exist_ok=True)

        if self.use_lora:
            adapter_path = base_path / 'adapter'
            if not adapter_path.exists():
                os.makedirs(adapter_path)
            assert adapter_params is not None
            torch.save(adapter_params, adapter_path / 'adapter_weights.pth')
        else:
            adapter_path = base_path / 'adapter'
            if not adapter_path.exists():
                os.makedirs(adapter_path)
            torch.save(adapter_params, adapter_path / 'adapter_weights.pth')
            #self.model.save(model_path)

        save_dict = {
            'optimizer': optimizer_params or self.optimizer.state_dict(),
            'iter_num': self.iter_num,
        }
        if self.lr_scheduler is not None:
            save_dict['lr_scheduler'] = self.lr_scheduler.state_dict()

        meta_info = self.meta_info
        with open(base_path / 'meta_info.json', 'w') as f:
            json.dump(meta_info, f)
        torch.save(save_dict, base_path / 'trainer_info.pth')

    @staticmethod
    def load(base_path: Path,
             model: LanguageModels,
             optimizer: torch.optim.Optimizer,
             tokenizer: AutoTokenizer,
             data_loader: torch.utils.data.DataLoader,
             lr_scheduler,
             rank: int = 0,
             local_rank: int = 0,
             world_size: int = 1,
             iter_num: int =0,
             use_lora: bool = False,
             device: torch.device = torch.device('cuda'),
             new_path: Path | None = None, **additional_kwargs):
        with open(base_path / 'meta_info.json', ) as f:
            meta_info = json.load(f)
        meta_info['optimizer_args'] = OptimizerArgs(**meta_info['optimizer_args'])
        meta_info['model_path'] = new_path or base_path
        if additional_kwargs is not None:
            meta_info.update(additional_kwargs)
        meta_info['model_path'] = new_path or base_path

        meta_info['rank'] = rank
        meta_info['local_rank'] = local_rank
        meta_info['world_size'] = world_size
        """
        if not use_lora:
            model.load_state_dict(torch.load(
                base_path / 'model' / 'model_weights.pth', map_location=device
            ))
        else:
            model.load_adapter(base_path / 'adapter',)

        trainer_info = torch.load(base_path / 'trainer_info.pth', map_location=device, weights_only=True)

        optimizer.load_state_dict(trainer_info['optimizer'])

        fp16_type = check_fp16_dtype()
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16 if fp16_type == 'bfloat16' else torch.float16,
            reduce_dtype=torch.bfloat16 if fp16_type == 'bfloat16' else torch.float16,
        )
        apply_activation_checkpointing(model)

        mesh = DeviceMesh(device_type='cuda', mesh=[i for i in range(world_size)])
        apply_fsdp(
            model,
            mesh,
            mp_policy,
            modules_to_shard={TransformerBlock, },
        )

        if meta_info['lr_scheduler_type'] == 'LambdaLR':
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: min(
                    1,
                    max((step + 1) / (meta_info['eval_iters'] // 5), 0.1),
                    max((meta_info['eval_iters'] - step) / (meta_info['eval_iters'] // 5), 0.1),
                ),
            )
        else:
            lr_scheduler = None

        lr_scheduler.load_state_dict(trainer_info['lr_scheduler'])
        """

        trainer = HFLongBenchDistillTrainer(model, optimizer, tokenizer=tokenizer,
                                         lr_scheduler=lr_scheduler,
                                         data_loader=data_loader, **meta_info)

        trainer.iter_num = iter_num
        return trainer

    def get_lr(self, it):
        return self.lr_max
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.lr_max * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.lr_min
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.lr_min + coeff * (self.lr_max - self.lr_min)

    def train(self):
        self.model.train()
        while self.iter_num <= self.max_iters:
            local_step = 0
            for ste, batch in enumerate(tqdm(self.train_loader)):
                batch = {k: v.to(f"cuda:{self.local_rank}") for k, v in batch.items()}
                # we first get the raw model output
                input_ids = batch['input_ids']
                bsz = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                #seq_parallel_chunk_size = seq_len // self.world_size
                #seq_parallel_chunk_start = seq_parallel_chunk_size * self.rank
                #seq_parallel_chunk_end = seq_parallel_chunk_start + seq_parallel_chunk_size
                seq_parallel_chunk_start = 0
                seq_parallel_chunk_end = seq_len
                print(f'seq_len: {seq_len}')
                print(f'iter:{self.iter_num}')

                position_ids = torch.arange(
                    seq_parallel_chunk_start,
                    seq_parallel_chunk_end,
                    device=input_ids.device,
                ).unsqueeze(0)

                self.model.eval()
                for k, v in self.model.named_modules():
                    if hasattr(v, 'adapter_params'):
                        v.disabled = True

                with torch.no_grad():
                    original_hidden_states = self.model(
                        input_ids=input_ids[:, seq_parallel_chunk_start:seq_parallel_chunk_end],
                        position_ids=position_ids,
                        output_hidden_states=False,
                        output_attentions=False,
                    )[0]

                for k, v in self.model.named_modules():
                    if hasattr(v, 'adapter_params'):
                        v.disabled = False

                self.model.train()
                nats_hidden_states = self.model(
                    input_ids=input_ids[:, seq_parallel_chunk_start:seq_parallel_chunk_end],
                    position_ids=position_ids,
                    output_hidden_states=False,
                    output_attentions=False,
                )[0]

                labels = batch["labels"][:, seq_parallel_chunk_start:seq_parallel_chunk_end]

                if self.train_on_labels:
                    label_mask = labels > 0
                else:
                    label_mask = labels != -100

                num_labels = label_mask.sum()
                global_num_labels = num_labels.clone().detach()
                dist.all_reduce(global_num_labels)

                model_out_origin = original_hidden_states[label_mask].float()
                model_out_nats = nats_hidden_states[label_mask].float()
                distill_loss = (model_out_nats - model_out_origin).pow(2).mean(-1).sum() * self.world_size / num_labels


                raw_model_loss = None
                model_loss = None
                model_loss_labels=None
                raw_model_loss_labels=None
                if self.model_lm_head_weight is not None:
                    msk_paddings = labels != -100
                    original_hidden_states = original_hidden_states[msk_paddings].unsqueeze(0)
                    nats_hidden_states = nats_hidden_states[msk_paddings].unsqueeze(0)
                    input_ids_ = input_ids[msk_paddings].unsqueeze(0)

                    #torch.cuda.empty_cache()

                    self.model_lm_head_weight = self.model_lm_head_weight.to(original_hidden_states)
                    logits_origin = nn.functional.linear(original_hidden_states, self.model_lm_head_weight)
                    logits_nats = nn.functional.linear(nats_hidden_states, self.model_lm_head_weight)

                    #torch.cuda.empty_cache()

                    raw_model_loss = nn.functional.cross_entropy(
                        logits_origin[:,:-1].view(-1, logits_origin.size(-1)),
                        input_ids_[:, 1:].flatten(),
                        ignore_index=-100,
                    )

                    model_loss = nn.functional.cross_entropy(
                        logits_nats[:,:-1].view(-1, logits_nats.size(-1)),
                        input_ids_[:, 1:].flatten(),
                        ignore_index=-100,
                    )

                    pos_ids_ = position_ids
                    pos_ids_ = pos_ids_ - pos_ids_[:,0].clone()

                    value_ranges = pos_ids_[[labels > 0]][:-1]
                    logits_origin_train = logits_origin[:, value_ranges]
                    logits_nats_train = logits_nats[:, value_ranges]
                    targets_train = input_ids[:, 1 + value_ranges]

                    raw_model_loss_labels = nn.functional.cross_entropy(
                        logits_origin_train[:, :].view(-1, logits_origin_train.size(-1)),
                        targets_train[:, ].flatten(),
                        ignore_index=-100,
                    )

                    model_loss_labels = nn.functional.cross_entropy(
                        logits_nats_train[:, :].view(-1, logits_nats_train.size(-1)),
                        targets_train[:, :].flatten(),
                        ignore_index=-100,
                    )

                    dist.all_reduce(raw_model_loss, op=dist.ReduceOp.AVG)
                    dist.all_reduce(model_loss, op=dist.ReduceOp.AVG)

                    dist.all_reduce(raw_model_loss_labels, op=dist.ReduceOp.AVG)
                    dist.all_reduce(model_loss_labels, op=dist.ReduceOp.AVG)


                    print(f'model_loss: {model_loss}')
                    print(f'raw model loss: {raw_model_loss}')


                    print(f'model loss train labels: {model_loss_labels}')
                    print(f'raw model loss train labels: {raw_model_loss_labels}')
                    del logits_origin
                    del logits_nats
                    del logits_origin_train
                    del logits_nats_train
                    torch.cuda.empty_cache()

                distill_loss.backward()

                local_step = (local_step + 1) % self.gradient_accumulation_steps

                dist.all_reduce(distill_loss, op=dist.ReduceOp.AVG)

                if local_step != 0:
                    continue

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()

                print(f'loss: {distill_loss}')
                torch.cuda.empty_cache()

                self.iter_num += 1
                if self.rank == 0:
                    if self.log_wandb:
                        wandb_log_dict = {"distill_loss": distill_loss.item(),
                                          }
                        wandb_log_dict.update({f'msk_fraction_{i}':  layer.self_attn.sparse_size for i, layer in enumerate(self.model.layers)})
                        if raw_model_loss is not None:
                            wandb_log_dict.update({
                                "loss Transformer": raw_model_loss,
                                "model_loss": model_loss,
                                "model_loss_labels": model_loss_labels,
                                "model_loss_labels_raw": raw_model_loss_labels,
                            })
                        wandb.log(
                            wandb_log_dict,
                            step=self.iter_num,
                        )

                if self.iter_num % self.eval_iters == 0:
                    adapter_params = gather_cpu_state_dict(self.model,
                                                           self.rank == 0,
                                                           device=self.device,
                                                           adapter_weights_only=True, )
                    optimizer_params = training.get_full_optimizer_state_dict(
                        opt=self.optimizer,
                        is_rank_zero=self.rank == 0,
                        device=self.device
                    )
                    if self.rank == 0:
                        print('saving!!!')
                        """
                        losses = self.estimate_loss(seq_parallel_chunk_size)
                        print(f"step {self.iter_num}, block size {seq_parallel_chunk_size}:"
                              f" train loss {losses['train']:.4f}, val loss {losses:.4f}")
                        if self.log_wandb:
                            wandb.log({
                                f"regression loss {seq_parallel_chunk_size}": losses,
                                # "mfu": running_mfu * 100,  # convert to percentage
                            })
                        """

                        self.save(self.model_path, adapter_params, optimizer_params)
                if self.iter_num >= self.max_iters:
                    break
                    #dist.barrier()

            dist.barrier()
            adapter_params = gather_cpu_state_dict(self.model,
                                                   self.rank == 0,
                                                   device=self.device,
                                                   adapter_weights_only=True, )
            optimizer_params = training.get_full_optimizer_state_dict(
                opt=self.optimizer,
                is_rank_zero=self.rank == 0,
                device=self.device
                )
            self.save(self.model_path, adapter_params, optimizer_params)
            dist.barrier()
            return



