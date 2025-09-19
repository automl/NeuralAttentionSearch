from dataclasses import dataclass
import torch


@dataclass
class OptimizerArgs:
    type: str = 'adamw',
    lr: float = 1e-5,
    beta1: float = 0.9
    beta2: float = 0.95,
    weight_decay: float = 0.1


def get_optimizer(optimizer_args: OptimizerArgs, optim_groups:  list[dict]) -> torch.optim.Optimizer:
    if optimizer_args.type == 'adamw':
        optimizer = torch.optim.AdamW(
            params=optim_groups,
            lr=optimizer_args.lr,
            betas=(optimizer_args.beta1, optimizer_args.beta2),
        )
    elif optimizer_args.type == 'adam':
        optimizer = torch.optim.Adam(
            params=optim_groups,
            lr=optimizer_args.lr,
            betas=(optimizer_args.beta1, optimizer_args.beta2),
        )
    else:
        raise NotImplemented(f'Unknown optimizer type: {optimizer_args}')
    return optimizer
