import os
import random
from pathlib import Path
from functools import partial

import hydra
import numpy as np
import omegaconf
import tiktoken
import torch
import wandb
import pickle

from nats.models.architecture import LanguageModels
from nats.models.model_configuration import TransformerArgs, CacheArgs
from nats.training.optimizer import OptimizerArgs
from nats.training.multi_gpus import init_dist, multi_gpu_setup
from nats.training.data.shakespeare import get_shakespeare_dataset
from nats.training.data.dataset_util import process_tiktoken, process_hf, get_raw_datasets, tokenize_dataset, prepare_dataset
from nats.training.trainer.trainer import TransformerTrainer


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Otherwise, Conv1D with dilation will be too slow
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_vocab_size(dataset_dir: Path, tokenizer_name: str):
    meta_path = (dataset_dir / 'meta.pkl')
    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
    #elif cfg.tokenizer.model_name == 'gpt2':
    elif tokenizer_name == 'gpt2':
        vocab_size = 50304
    elif tokenizer_name == 'llama3':
        vocab_size = 128256
    else:
        raise NotImplementedError("Unknown vocab size for the tokenizer!")

    return vocab_size

dataset_names = {'pg19': 'pg19',
                 "openwebtext": "owt",
                 'shakespeare': 'shakespeare',
                 }


def train_model(cfg: omegaconf.DictConfig):
    # Use DDP for multi-gpu training
    if cfg.n_gpus > 1:
        dist_kwargs = dict()
        rank, local_rank, world_size = init_dist(**dist_kwargs)
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        local_rank = 0
        world_rank = 0

    seed = int(cfg.seed) + rank
    seed_everything(seed)

    cfg_transformer = cfg.transformer_args

    dataset_name_display = dataset_names[cfg.dataset.name]
    if cfg_transformer.nats_enable:
        sparse_idx = cfg_transformer.sparse_regularized_value
        local_seq_max_length = cfg_transformer.local_seq_max_length
        chunk_size = cfg_transformer.chunk_size
        chunk_merge_method = cfg_transformer.chunk_merge_method

        #model_name = f"nats_{str(sparse_idx).replace('-0', '')}_SWindowlen{local_seq_max_length}_chunk{chunk_size}_{chunk_merge_method}"
        #if cfg_transformer.nats_backward_compress_on_q:
        #    model_name = f"{model_name}_compress_q"
        task_name = f"{dataset_name_display}_NAtS"
    else:
        model_name = f'{dataset_name_display}_transformer'
        task_name = f'{dataset_name_display}_baseline'
    model_name = cfg.model.name

    if rank == 0:
        cfg.wandb.name = f'{model_name}'
        cfg.wandb.project = task_name
        wandb.init(**cfg.wandb)
    model_name = cfg.model.name

    # prepare dataset
    cfg_dataset = cfg.dataset

    if cfg_dataset.char_level:
        dataset_dir = Path(cfg_dataset.base_dir) / f'{cfg_dataset.name}_charlevel'
    else:
        dataset_dir = Path(cfg_dataset.base_dir) / cfg_dataset.name

    if not ((dataset_dir / 'train.bin').exists() or (dataset_dir / 'val.bin').exists()):
        prepare_dataset(cfg)

    vocab_size = get_vocab_size(dataset_dir, cfg.tokenizer.model_name)

    path_model = Path(cfg.model.base_dir) / model_name
    if path_model.exists():
        cfg_trainer = cfg.trainer
        trainer = TransformerTrainer.load(path_model,
                                          local_rank=local_rank,
                                          use_multi_gpus=cfg.n_gpus > 1,
                                          dataset_path=dataset_dir,
                                          gradient_accumulation_steps=cfg_trainer.gradient_accumulation_steps,
                                          log_wandb=cfg.wandb.mode != 'disabled',
                                          #device=torch.cuda.current_device(),
                                          is_master_process=rank == 0,
                                          on_cpu=cfg_trainer.on_cpu,
                                          save_after_eval=cfg_trainer.save_after_eval,
                                          )
        # this helps to fully reproduce the samples that we sampled so far
        #trainer.get_batch_virtual(trainer.iter_num - 1)
    else:
        # prepare models
        d_model = cfg.model_general.d_model
        use_multi_gpus = cfg.n_gpus > 1

        transformer_args = TransformerArgs(
            dim=d_model,
            n_layers=cfg_transformer.n_layers,
            n_heads=cfg_transformer.n_heads,
            n_kv_heads=cfg_transformer.n_kv_heads,
            n_msks=cfg_transformer.n_msks,
            multiple_of=cfg_transformer.multiple_of,
            ffn_dim_multiplier=cfg_transformer.ffn_dim_multiplier,
            norm_eps=cfg_transformer.norm_eps,
            rope_theta=cfg_transformer.rope_theta,
            apply_pos_emb=cfg_transformer.apply_pos_emb,
            nats_enable=cfg_transformer.nats_enable,
            sparse_regularized_value=cfg_transformer.sparse_regularized_value,
            local_seq_max_length=cfg_transformer.local_seq_max_length,
            chunk_size=cfg_transformer.chunk_size,
            compress_on_q=cfg_transformer.nats_backward_compress_on_q,
            chunk_merge_method=cfg_transformer.chunk_merge_method,
            vocab_size=vocab_size,
            on_ddp=use_multi_gpus
        )

        # cache
        cache_args = CacheArgs(
            n_kv_heads=cfg_transformer.n_kv_heads,
            n_msks=cfg_transformer.n_msks,
        )

        model = LanguageModels(tokenizer_vocab_size=vocab_size, transformer_args=transformer_args,
                               cache_args=cache_args,)

        optimizer_args = OptimizerArgs(
            **cfg.optimizer_args
        )
        optimizer = model.configurate_optimizer(optimizer_args)

        # setup multi gpus
        cfg_trainer = cfg.trainer
        model, module = multi_gpu_setup(model, rank, use_multi_gpus, on_cpu=cfg_trainer.on_cpu)

        trainer = TransformerTrainer(
            model, module, optimizer, optimizer_args, dataset_path=dataset_dir,
            model_path=path_model, block_size_train=cfg_trainer.block_size_train,
            block_size_eval=list(cfg_trainer.block_size_eval),
            gradient_accumulation_steps=cfg_trainer.gradient_accumulation_steps,
            batch_size=cfg_trainer.batch_size,
            grad_clip=cfg_trainer.grad_clip, warmup_iters=cfg_trainer.warmup_iters,
            lr_decay_iters=cfg_trainer.lr_decay_iters, max_iters=cfg_trainer.max_iters,
            eval_iters=cfg_trainer.eval_iters, eval_interval=cfg_trainer.eval_interval,
            lr_max=optimizer_args.lr, lr_min=cfg_trainer.lr_min, use_multi_gpus=use_multi_gpus,
            is_master_process=rank == 0, log_wandb=cfg.wandb.mode != 'disabled',
            on_cpu=cfg_trainer.on_cpu,
            save_after_eval=cfg_trainer.save_after_eval,
        )
    trainer.train()


@hydra.main(config_path="configs", config_name="base.yaml")
def main(cfg: omegaconf.DictConfig):
    # Multi-gpu training
    if cfg.n_gpus > 1:
        train_model(cfg)
    else:
        with omegaconf.open_dict(cfg):
            cfg.rank = 0
            cfg.local_rank = 0
            cfg.world_size = 1
        train_model(cfg)


if __name__ == '__main__':
    main()
