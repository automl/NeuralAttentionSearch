# the main part of this function is adapted from duoattention with MIT license:
# https://github.com/mit-han-lab/duo-attention/blob/main/duo_attn/train.py

import os
import random
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import torch
import wandb
from transformers import AutoTokenizer
import json

import torch.distributed as dist
from torchtune import training

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.llama import LlamaForCausalLM
from transformers.models.mistral import MistralForCausalLM

from nats.utils import check_fp16_dtype
from nats.models.model_configuration import TransformerArgs, CacheArgs
from nats.training.optimizer import OptimizerArgs, get_optimizer
from nats.training.multi_gpus import init_dist, multi_gpu_setup
from nats.training.data.dataset_util import get_raw_datasets
from nats.training.trainer.hf_distill_long_bench_trainer import HFLongBenchDistillTrainer

from nats.models.transformer.hf.llama import enable_llama_nats_training
from nats.models.transformer.hf.llama_nats_chunk import enable_llama_nats_chunk_training
from nats.models.transformer.hf.mistral import enable_mistral_nats_training
from nats.models.transformer.hf.mistral_nats_chunk import enable_mistral_nats_chunk_training
from nats.models.transformer.hf.lora import config_lora_for_model, get_fine_tune_models
from experiments.finetune_datasets.long_bench_dataset import LongBenchDataset, get_dataset, get_supervised_dataloader

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from experiments.finetune_datasets.duo_attn_dataset import (
    get_dataset,
    MultiplePasskeyRetrievalDataset,
    get_supervised_dataloader
)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Otherwise, Conv1D with dilation will be too slow
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_fsdp(model: torch.nn.Module, mesh, mp_policy, modules_to_shard):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    fsdp_config = {"mp_policy": mp_policy, "mesh": mesh, "reshard_after_forward": True}

    for module in model.modules():
        if any([isinstance(module, m) for m in modules_to_shard]):
            fully_shard(module, **fsdp_config)
    fully_shard(model, **fsdp_config)


def tune_model(cfg: omegaconf.DictConfig):
    # Use DDP for multi-gpu training
    if cfg.n_gpus > 1:
        dist_kwargs = dict()
        rank, local_rank, world_size = init_dist(**dist_kwargs)
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    seed = int(cfg.seed) + rank
    seed_everything(seed)

    base_model_name = cfg.model.pre_trained_model_name

    sparse_idx = cfg.transformer_args.sparse_regularized_value
    lr = cfg.optimizer_args.lr
    wd = cfg.optimizer_args.weight_decay
    local_seq_max_length = cfg.transformer_args.local_seq_max_length
    use_lora = cfg.trainer.use_lora
    chunk_size = cfg.transformer_args.chunk_size

    task_name = f"nats_{str(sparse_idx).replace('-0', '')}_SWindowlen{local_seq_max_length}_lr{str(lr).replace('-0', '')}_wd{str(str(wd).replace('-0', ''))}"
    if chunk_size > 1:
        task_name = f"{task_name}_chunk{chunk_size}"

    if use_lora:
        task_name = f'{task_name}_useLora'
        if cfg.trainer.tune_par is not None:
            task_name = f'{task_name}_LORAModule:{cfg.trainer.tune_par}'

    if cfg.trainer.train_on_labels:
        task_name = f'{task_name}_OnLabels'

    task_name = f'{task_name}_train{str(cfg.trainer.max_iters // 1000)}'

    if cfg.dataset.with_prompt:
        task_name = f'{task_name}_withprompt'

    task_name = f'{task_name}_mixed'

    if rank == 0:
        cfg.wandb.name = f'{base_model_name}_{task_name}'
        cfg.wandb.project = f'finetune_{base_model_name}_{cfg.dataset.source}'
        wandb.init(**cfg.wandb)

    # prepare dataset
    cfg_dataset = cfg.dataset
    pre_trained_model_path = str(Path(cfg.model.pre_trained_model_path) / base_model_name)

    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_path)

    if 'Llama-3' in base_model_name:
        tokenizer.add_special_tokens({'pad_token': "<|reserved_special_token_247|>"})
    elif 'Mistral' in base_model_name:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    else:
        pass
        #raise ValueError(base_model_name)

    dataset_dir = Path(cfg_dataset.base_dir) / cfg_dataset.name

    cfg_dataset = cfg.dataset

    dataset_path = cfg_dataset.dataset_path

    if cfg_dataset.source == 'LongBench':
        train_dataset = LongBenchDataset(
        dataset_path=Path(dataset_path),
        tokenizer=tokenizer,
        max_length=cfg_dataset.context_length_max,
        pad_to_multiple_of=128,
        with_prompt=cfg.dataset.with_prompt,
        n_datas=cfg.trainer.max_iters,
        cfg_dataset=cfg.dataset
        )

        train_dataloader = get_supervised_dataloader(
            train_dataset, tokenizer, cfg.trainer.batch_size, shuffle=True
        )
    elif cfg_dataset.source == 'Synthetic':
        haystack_dataset = get_dataset(cfg_dataset.name, cfg_dataset.dataset_path, split="train")

        train_dataset = MultiplePasskeyRetrievalDataset(
            haystack_dataset,
            tokenizer,
            max_length=cfg_dataset.context_length_max,
            min_depth_ratio=cfg_dataset.min_needle_depth_ratio,
            max_depth_ratio=cfg_dataset.max_needle_depth_ratio,
            context_length_min=cfg_dataset.context_length_min,
            context_length_max=cfg_dataset.context_length_max,
            context_lengths_num_intervals=cfg_dataset.context_lengths_num_intervals,
            depth_ratio_num_intervals=cfg_dataset.depth_ratio_num_intervals,
            num_passkeys=cfg_dataset.num_passkeys,
            pad_to_multiple_of=128,
        )

        train_dataloader = get_supervised_dataloader(
            train_dataset, tokenizer, cfg.trainer.batch_size, shuffle=True
        )
    else:
        raise NotImplementedError(f"{cfg_dataset.source}")

    cfg_trainer = cfg.trainer

    path_model = Path(cfg.model.base_dir) / base_model_name / task_name

    d_model = cfg.model_general.d_model

    cfg_transformer = cfg.transformer_args
    use_multi_gpus = cfg.n_gpus > 1

    transformer_args = TransformerArgs(
        dim=d_model,
        nats_enable=cfg_transformer.nats_enable,
        n_msks=cfg_transformer.n_msks,
        proj_layer_is_ssm=cfg_transformer.proj_layer_is_ssm,
        on_ddp=use_multi_gpus,
        chunk_size=cfg_transformer.chunk_size,
        sparse_regularized_value=cfg_transformer.sparse_regularized_value,
        local_seq_max_length=cfg_transformer.local_seq_max_length
    )
    if 'Llama-3' in base_model_name:
        model = LlamaForCausalLM.from_pretrained(pre_trained_model_path)
        #if transformer_args.chunk_size > 1:
        if True:
            # we always use the chunk setting
            enable_llama_nats_chunk_training(model, transformer_args)
        else:
            enable_llama_nats_training(model, transformer_args)
        model_lm_head_weight = model.lm_head.weight.data
        model = model.model
    elif 'Mistral' in base_model_name:
        model = MistralForCausalLM.from_pretrained(pre_trained_model_path)
        model.resize_token_embeddings(len(tokenizer))
        #enable_mistral_nats_training(model, transformer_args)
        enable_mistral_nats_chunk_training(model, transformer_args)
        model_lm_head_weight = model.lm_head.weight.data
        model = model.model
    else:
        raise ValueError(base_model_name)

    if cfg_trainer.use_lora:
        involved_modules = get_fine_tune_models(str(cfg.trainer.tune_par))

        config_lora_for_model(model, involved_modules=involved_modules)

    if (path_model / 'meta_info.json').exists():
        adapter_sd = torch.load(path_model / 'adapter' / 'adapter_weights.pth')
        model.load_state_dict(adapter_sd, strict=False)

    torch.cuda.set_device(local_rank)
    model = model.to(local_rank)
    model_lm_head_weight = model_lm_head_weight.to(local_rank)

    if world_size > 1:
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
            modules_to_shard={LlamaDecoderLayer, MistralDecoderLayer},
        )

    optimizer_args = OptimizerArgs(
        **cfg.optimizer_args
    )

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=optimizer_args.lr,
                                  betas=(optimizer_args.beta1, optimizer_args.beta2), )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            1,
            max((step + 1) / (cfg_trainer.max_iters // 5), 0.1),
            max((cfg_trainer.max_iters - step) / (cfg_trainer.max_iters // 5), 0.1),
        ),
    ) if cfg_trainer.lr_scheduler_type == 'LambdaLR' else None

    if (path_model / 'meta_info.json').exists():
        device = torch.device(type="cuda", index=torch.cuda.current_device())
        trainer_info = torch.load(path_model / 'trainer_info.pth', map_location=device
                                  , weights_only=True)
        training.load_from_full_optimizer_state_dict(optimizer, trainer_info['optimizer'],
                                                     device=device,
                                                     )
        #optimizer.load_state_dict(trainer_info['optimizer'])
        lr_scheduler.load_state_dict(trainer_info['lr_scheduler'])

        cfg_trainer = cfg.trainer
        trainer = HFLongBenchDistillTrainer.load(path_model,
                                              model=model,
                                              optimizer=optimizer,
                                              tokenizer=tokenizer,
                                              data_loader=train_dataloader,
                                              lr_scheduler=lr_scheduler,
                                              rank=rank,
                                              local_rank=local_rank,
                                              world_size=world_size,
                                              use_lora=True,
                                              gradient_accumulation_steps=cfg_trainer.gradient_accumulation_steps,
                                              log_wandb=cfg.wandb.mode != 'disabled',
                                              on_cpu=cfg_trainer.on_cpu,
                                              save_after_eval=cfg_trainer.save_after_eval,
                                        model_lm_head_weight=model_lm_head_weight,
                                              )
    else:

        trainer = HFLongBenchDistillTrainer(
            model, optimizer, optimizer_args, tokenizer=tokenizer,
            data_loader=train_dataloader,
            model_lm_head_weight=model_lm_head_weight,
            lr_scheduler=lr_scheduler,
            dataset_path=dataset_dir,
            model_path=path_model,
            gradient_accumulation_steps=cfg_trainer.gradient_accumulation_steps,
            batch_size=cfg_trainer.batch_size,
            grad_clip=cfg_trainer.grad_clip,
            eval_iters=cfg_trainer.eval_iters, eval_interval=cfg_trainer.eval_interval,
            lr_max=optimizer_args.lr,
            #lr_min=cfg_trainer.lr_min,
            max_iters=cfg_trainer.max_iters,
            use_multi_gpus=use_multi_gpus,
            use_lora=cfg_trainer.use_lora,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            log_wandb=cfg.wandb.mode != 'disabled',
            on_cpu=cfg_trainer.on_cpu,
            save_after_eval=cfg_trainer.save_after_eval,
            train_on_labels=cfg_trainer.train_on_labels
        )
    trainer.train()
    dist.barrier()


@hydra.main(config_path="configs", config_name="finetune_distill.yaml")
def main(cfg: omegaconf.DictConfig):
    # Multi-gpu training
    if cfg.n_gpus > 1:
        tune_model(cfg)
    else:
        with omegaconf.open_dict(cfg):
            cfg.rank = 0
            cfg.local_rank = 0
            cfg.world_size = 0
        tune_model(cfg)



if __name__ == '__main__':
    main()
