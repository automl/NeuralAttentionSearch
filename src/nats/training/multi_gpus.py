import sys
import builtins
import wandb
import os
import torch.distributed as dist
import torch
import socket
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

def suppress_print():
    """Suppresses printing from the current process."""
    def ignore(*_objects, _sep=" ", _end="\n", _file=sys.stdout, _flush=False):
        pass
    builtins.print = ignore

def suppress_wandb():
    """Suppresses wandb logging from the current_process."""
    def ignore(data, step=None, commit=None, sync=None):
        pass
    wandb.log = ignore


def init_dist(back_end="nccl"):
    """This function is slurm only!"""
    WORLD_SIZE = int(os.environ["SLURM_NTASKS"])
    world_rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    NODELIST = os.environ['SLURM_NODELIST']
    NGPUS_PER_NODE = torch.cuda.device_count()
    HOSTNAME = socket.gethostname()
    print("Rank", world_rank, " runs on node" ,HOSTNAME, " and uses GPU", local_rank, "World SIZE", WORLD_SIZE, "NGPUS_PER_NODE", NGPUS_PER_NODE)

    dist.init_process_group(backend=back_end, rank=world_rank, world_size=WORLD_SIZE)
    print("INIT FINISHED!!!")

    torch.distributed.barrier()

    #if not model_parallel_is_initialized():
    #    model_parallel_size = WORLD_SIZE
    #    initialize_model_parallel(model_parallel_size)

    if local_rank != 0:
        suppress_print()
        suppress_wandb()
    return local_rank, world_rank, WORLD_SIZE


def multi_gpu_setup(model: torch.nn.Module, local_rank, use_multi_gpus: bool = False, on_cpu: bool=False,
                    model_is_peft: bool = False):
    """This function is used to prepare the """
    if use_multi_gpus:
        device = torch.cuda.current_device()
        model.device = device
        if model_is_peft:
            # TODO there should be better way to handle this
            model.base_model.model.device = device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
            # static_graph=True
        )
        #model.configure_optimizers = model.module.configure_optimizers
        module = model.module

        return model, module
    else:
        if on_cpu:
            device = torch.device('cpu')
        else:
            device = torch.cuda.current_device()

        model.device = device
        if model_is_peft:
            model.base_model.model.device = device
        return model, model


def apply_fsdp(model: torch.nn.Module, mesh, mp_policy, modules_to_shard):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    fsdp_config = {"mp_policy": mp_policy, "mesh": mesh, "reshard_after_forward": True}

    for module in model.modules():
        if any([isinstance(module, m) for m in modules_to_shard]):
            fully_shard(module, **fsdp_config)
    fully_shard(model, **fsdp_config)

