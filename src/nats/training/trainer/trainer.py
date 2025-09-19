"""
The functions here are mostly adapted from
https://github.com/karpathy/nanoGPT/blob/master/train.py
"""
import json
import dataclasses
import os
import time
from pathlib import Path
import numpy as np

import torch
import math
import wandb
from dataclasses import asdict, dataclass
from dataclasses import dataclass
from torch import nn
from tqdm import tqdm

from nats.utils import get_kwargs
from nats.training.multi_gpus import multi_gpu_setup
from nats.training.optimizer import OptimizerArgs
from nats.models.architecture import LanguageModels


class TransformerTrainer:
    def __init__(self,
                 model: nn.parallel.DistributedDataParallel,
                 module: LanguageModels,
                 optimizer: torch.optim.Optimizer,
                 optimizer_args: OptimizerArgs,
                 dataset_path: Path,
                 model_path: Path,
                 block_size_train: int = 1024,
                 block_size_eval: list[int] | None = None,
                 gradient_accumulation_steps: int = 1,
                 batch_size: int = 64,
                 grad_clip: float = 1.0,
                 warmup_iters: int = 2000,
                 lr_decay_iters: int = 600000,  # should be ~= max_iters per Chinchilla
                 max_iters: int = 600000,
                 eval_iters: int = 200,
                 eval_interval: int = 2000,
                 lr_max: float = 6e-4,  # max learning rate
                 lr_min: float = 6e-5,  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
                 use_multi_gpus: bool = False,
                 is_master_process: bool = True,
                 iter_number: int = 0,
                 log_wandb: bool = True,
                 eval_only: bool = False,
                 on_cpu: bool = False,
                 save_after_eval: bool = True,
                 best_val_loss: float = 1e9,
                 ):
        all_kwargs = get_kwargs()
        self.optimizer = optimizer

        self.model = model
        self.module = module

        self.is_master_process = is_master_process

        self.block_size_train = block_size_train
        if block_size_eval is None:
            block_size_eval = [1024, 2048]
        self.block_size_eval = block_size_eval

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
            self.device = torch.cuda.current_device()
        self.module.device = self.device

        self.batch_size = batch_size
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)
        self.opt_model_path = Path(str(model_path) + '_opt')

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

    def save(self, base_path: Path | str):
        if not isinstance(base_path, Path):
            base_path = Path(base_path)
        if not base_path.exists():
            os.makedirs(base_path, exist_ok=True)
        model_path = base_path / 'model'
        if not model_path.exists():
            os.makedirs(model_path, exist_ok=True)

        self.module.save(model_path)

        save_dict = {
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
        }

        meta_info = self.meta_info
        with open(base_path / 'meta_info.json', 'w') as f:
            json.dump(meta_info, f)
        torch.save(save_dict, base_path / 'trainer_info.pth')

    @staticmethod
    def load(base_path: Path, local_rank: int = 0, use_multi_gpus: bool = False, device=torch.device('cuda'),
             new_path: Path | None = None, **additional_kwargs):
        with open(base_path / 'meta_info.json', ) as f:
            meta_info = json.load(f)
        meta_info['optimizer_args'] = OptimizerArgs(**meta_info['optimizer_args'])
        model = LanguageModels.load(base_path / 'model', device=device, )
        optimizer = model.configurate_optimizer(meta_info['optimizer_args'])
        meta_info['model_path'] = new_path or base_path
        if additional_kwargs is not None:
            meta_info.update(additional_kwargs)

        trainer_info = torch.load(base_path / 'trainer_info.pth', map_location=device, weights_only=True)

        optimizer.load_state_dict(trainer_info['optimizer'])

        model, module = multi_gpu_setup(model, local_rank, use_multi_gpus)
        meta_info['model_path'] = new_path or base_path

        trainer = TransformerTrainer(model, module, optimizer, **meta_info)

        if 'scaler' in trainer_info:
            trainer.scaler.load_state_dict(trainer_info['scaler'])
        trainer.iter_num = trainer_info['iter_num']
        trainer.best_val_loss = trainer_info.get('best_val_loss', 1e9)
        return trainer

    def get_batch(self, split: str, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(str(self.dataset_path / 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(str(self.dataset_path / 'val.bin'), dtype=np.uint16, mode='r')

        ix = torch.randint(len(data) - block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((np.asarray(data[i:i + block_size])).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((np.asarray(data[i + 1:i + 1 + block_size])).astype(np.int64)) for i in ix])

        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x, y

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.lr_max * it / self.warmup_iters, 1.0
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.lr_min, 1.0
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.lr_min + coeff * (self.lr_max - self.lr_min), coeff

    def get_batch_virtual(self, iter_number: int,):
        """This function is used to virtually mimic the torch randint process such that we get similar data as we
        continue training our appproach"""
        block_size_train = self.block_size_train
        data_train = np.memmap(str(self.dataset_path / 'train.bin'), dtype=np.uint16, mode='r')
        len_train = len(data_train)
        del data_train
        data_val = np.memmap(str(self.dataset_path / 'val.bin'), dtype=np.uint16, mode='r')
        len_val = len(data_val)
        del data_val

        _ = torch.randint(len_train - block_size_train, (self.batch_size,))
        for i in range(iter_number):
            _ = torch.randint(len_train - block_size_train, (self.batch_size,))
            # mimic evaluation
            if i % self.eval_interval == 0 and self.is_master_process:
                for split in ['train', 'val']:
                    for block_size_eval in self.block_size_eval:
                        if split == 'train':
                            _ = torch.randint(len_train - block_size_eval, (self.batch_size,))
                        else:
                            _ = torch.randint(len_val - block_size_eval, (self.batch_size,))

            # mimic training
            for micro_step in range(self.gradient_accumulation_steps):
                _ = torch.randint(len_train - block_size_train, (self.batch_size,))

    def train(self):
        X, Y = self.get_batch('train', self.block_size_train)
        while True:
            # evaluate the loss on train/val sets and write checkpoints
            if self.iter_num % self.eval_interval == 0 and self.is_master_process:
                #"""
                val_loss = self.best_val_loss
                for it_eval, block_size in enumerate(self.block_size_eval):
                    #losses = self.estimate_loss_ar(block_size)
                    losses = self.estimate_loss(block_size)
                    print(f"step {self.iter_num}, block size {block_size}:"
                          f" train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                    if self.log_wandb:
                        wandb.log({
                            "block_size": block_size,
                            f"train/loss {block_size}": losses['train'],
                            f"val/loss {block_size}": losses['val'],
                            # "mfu": running_mfu * 100,  # convert to percentage
                        })
                    if it_eval == 0:
                        val_loss = losses['val']
                if self.is_master_process and self.save_after_eval and self.iter_num != 0:
                    self.save(base_path=self.model_path)
                    pass
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save(base_path=self.opt_model_path)
            self.iter_num += 1

            X, Y = self.train_iteration(iter_num=self.iter_num, X=X, Y=Y)

            # termination conditions
            if self.iter_num > self.max_iters:
                break

    @torch.no_grad()
    def estimate_loss(self, block_size: int):
        out = {}
        #self.module.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in tqdm(range(self.eval_iters)):
                X, Y = self.get_batch(split, block_size)
                with torch.cuda.amp.autocast(enabled=True, dtype=self.ptdtype):
                    logits, loss = self.model(X, Y, on_val=True)

                # self.module.reset_cache()
                losses[k] = loss.item()
            out[split] = losses.mean()
            # we need to clear the cache here.
        self.module.train()
        return out

    @torch.inference_mode()
    def estimate_loss_ar(self, block_size: int):
        out = {}
        self.module.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split, block_size)
                len_seq = X.shape[1]
                # TODO check if we could make this enabled as True
                with torch.cuda.amp.autocast(enabled=False, dtype=self.ptdtype):
                    loss_iter = torch.zeros(len_seq, device=X.device)
                    for i in range(len_seq):
                        logits, loss_iter[i] = self.model(X[:, [i]], Y[:, [i]], start_pos=i)
                loss = loss_iter.mean()

                losses[k] = loss.item()
                self.module.reset_cache()

            out[split] = losses.mean()
            # we need to clear the cache here.
        self.module.train()
        return out

    def train_iteration(self, iter_num, X: torch.Tensor | None, Y: torch.Tensor | None):
        # determine and set the learning rate for this iteration
        lr, coeff = self.get_lr(iter_num)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(self.gradient_accumulation_steps):
            if self.use_multi_gpus:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                self.model.require_backward_grad_sync = (micro_step == self.gradient_accumulation_steps - 1)
            X = X.to(self.device)
            Y = Y.to(self.device)
            with torch.cuda.amp.autocast(enabled=True, dtype=self.ptdtype):
                logits, loss1 = self.model(X, Y, coeff=coeff)
                loss = loss1.mean() / self.gradient_accumulation_steps  # scale the loss to account for gradient accumulation

            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = self.get_batch('train', self.block_size_train)
            # backward pass, with gradient scaling if training in fp16
            self.scaler.scale(loss).backward()

        # clip the gradient
        if self.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            #gradient_norm = self.module.grad_norm()

            if self.log_wandb:
                gradient_norm = self.module.grad_norm()
                wandb.log({'gradient_norm_weights': gradient_norm})
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        if self.log_wandb:
            wandb.log({
                "iter": iter_num,
                "train/loss": loss,
                "lr": lr,
                # "mfu": running_mfu * 100,  # convert to percentage
            })

        # step the optimizer and scaler if training in fp16
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # flush the gradients as soon as we can, no need for this memory anymore
        self.optimizer.zero_grad(set_to_none=True)
        return X, Y
