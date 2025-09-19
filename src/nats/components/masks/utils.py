from typing import Callable
import torch

import torch.nn.functional as F

from nats.components.masks.triton import cummax_reverse, construct_soft_mask


def inverse_cum_func(cum_func: Callable, input_tensor: torch.Tensor, len_seqs: int):
    reversed_range = torch.arange(len_seqs - 1, -1, -1)
    cum_res = cum_func(input_tensor[..., reversed_range], 1)[0][..., reversed_range]
    return cum_res


def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.,
                   return_soft: bool = True,
                   threshold: float = 0.5,
                   on_inference: bool = False,
                   ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    Assuming that we have a logit, then its corresponding probability is prob = Sigmoid(logit)
    if we apply gumble softmax to [prob, 1-prob], the gubmble softmax for logit is:
    exp((log(prob) + g1)/tau) / (exp((log(prob) + g1)/tau) + exp((log(1-prob) + g2)/tau))
    = Sigmoid(g1-g2+log(prob/1-prob))
    while log(prob/(1-prob)) = log(exp(logit)) = logit

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
      threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0
      on_inference: if it is on inference time, if it is in infrence mode, we do not need to generate random samples

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    if on_inference:
        gumbels = logits / tau
    else:
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = gumbels1 - gumbels2
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)#
    y_soft = F.sigmoid(gumbels)

    # Straight through.
    index = y_soft > threshold
    #y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format) + index
    y_hard = index.to(y_soft)
    ret_hard = y_hard - y_soft.detach() + y_soft
    # y_soft = torch.lerp(y_soft, ret_hard, ret_hard)
    # y_soft = torch.lerp(torch.full_like(ret_hard, fill_value=1e-8), ret_hard, ret_hard)
    if return_soft:
        return y_soft, ret_hard
    return ret_hard

import time


def generate_soft_seq_masks(end_seqs_soft: torch.Tensor,
                            end_seqs_hard: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
    end_seqs_idx, collected_soft_values = cummax_reverse(end_seqs_hard, end_seqs_soft)
    mask_soft = construct_soft_mask(end_seqs_idx=end_seqs_idx,
                                    end_seqs_soft=end_seqs_soft,
                                    end_seqs_soft_collected=collected_soft_values,
                                    start_idx=start_idx
                                    )
    return mask_soft


def generate_hard_seq_masks(end_seqs_hard: torch.Tensor, start_idx: int = 0,) -> torch.Tensor:
    """
    This function is similar to generate_soft_seq_masks, However, it only generates the bool masks values
    """
    seqlen = end_seqs_hard.shape[-1]
    end_seqs_hard[..., -1] = 1.
    idx_sub_seq = seqlen - 1 - torch.flip(torch.cummax(torch.flip(end_seqs_hard, (-1,)), -1)[1], (-1,))

    idx_sub_seq_ = idx_sub_seq.unsqueeze(-2)
    idx_ranges = torch.arange(start_idx, idx_sub_seq.shape[-1], device=idx_sub_seq.device).unsqueeze(-1)

    end_seq_mask = ((idx_sub_seq_ - idx_ranges) >= 0) | end_seqs_hard.unsqueeze(-2).bool()

    return end_seq_mask, idx_sub_seq
