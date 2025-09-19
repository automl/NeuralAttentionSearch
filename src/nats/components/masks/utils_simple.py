# The functions here are the pure torch implementation for the functions with the same nam under utils.py

import torch


def mark_sub_sequences(end_seqs_hard: torch.Tensor, preseverse_last_block: bool = False) -> torch.Tensor | tuple[
    torch.Tensor]:
    """
    This function is used to mark the item of each sub-sequence. Each sub-sequence is marked with the position of their
    end-seq values. If the last element from end_seqs_hard is False, then the elements in the last sub-sequences will
    be marked with the last index of this sequence
    For instance, if we have an end_seqs_hard with
    [F, F, T, F, F, T, F, F]
    Then the resulting tensor should be
    [2, 2, 2, 6, 6, 6, 8, 8]

    Args:
        end_seqs_hard (torch.Tensor): a torch tensor with shape [bsz, nh, n_ctx]. This value only contains bool values
            and shows if the current sub-sequence ends at the current location
    """
    len_seqs = end_seqs_hard.shape[-1]
    # TODO check if this version is faster than torch.where on gpus!
    seq_idx = torch.arange(len_seqs, device=end_seqs_hard.device)
    ones_indices = (
        seq_idx * end_seqs_hard + (len_seqs - 1) * (1 - end_seqs_hard)
    ).long()
    # ones_indices = torch.where(end_seqs_hard == 1, torch.arange(len_seqs, device=end_seqs_hard.device), len_seqs - 1)

    ones_indices[..., -1] = len_seqs - 1
    # torch does not support cummin reversely, we need to do that manually
    reversed_range = torch.arange(len_seqs - 1, -1, -1, device=ones_indices.device)
    idx_sub_seq = torch.cummin(ones_indices[..., reversed_range], -1)[0][..., reversed_range]

    if preseverse_last_block:
        is_last_block = (idx_sub_seq == (len_seqs - 1)).long()
        idx_sub_seq = is_last_block * seq_idx + (1 - is_last_block) * idx_sub_seq
        return idx_sub_seq, is_last_block

    return idx_sub_seq


def generate_seq_mask(idx_sub_seq: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
    """
    This function is used to generate a sequential masks that used for both training and inference.

    Args:
        idx_sub_seq (torch.Tensor): a tensor with shape [bsz, nh, n_ctx], a tensor recording to which subsequence the
            current token belongs
        start_idx (int): sometimes we might not want the full attention map. This value is used to control the size of
            the generated masks
    """
    # we apply mask to the sequence. Looking deeper into the generated mask
    # This requires us to have the following masks:

    #   T  -inf -inf | -inf -inf | -inf | -inf
    #   T    T  -inf | -inf -inf | -inf | -inf
    #   T    T    T  | -inf -inf | -inf | -inf
    # ----------------------------------------
    #   F    F  T(F) |   T  -inf | -inf | -inf
    #   F    F  T(F) |   T  T(F) | -inf | -inf
    #   F    F  T(F) |   F  T(F) |  T   | -inf
    # ----------------------------------------
    #   F    F  T(F) |   F  T(F) |  T   |   T
    # ----------------------------------------
    # -inf -inf   0  | -inf   0  |  0   |   0
    #   F    F    T  |   F    T  |  T   |   T
    #   3    3    3  |   5    5  |  7   |   7
    # ----------------------------------------
    #   1    2    3  |   4    5  |  7   |   7

    # We could split the entire mask with the 1.0 values within end_seqs, then the amount of "F" values with in each
    # sub-sequence equals to the cumulative amount of 0 + 1 within each sub-sequences

    # This result us a sequence with [3, 3, 3, 5, 5, 6, 7]
    idx_sub_seq_ = idx_sub_seq.unsqueeze(-2)
    idx_ranges = torch.arange(start_idx, idx_sub_seq.shape[-1], device=idx_sub_seq.device).unsqueeze(-1)
    end_seq_mask = (idx_sub_seq_ - idx_ranges) >= 0
    return end_seq_mask


def generate_soft_seq_masks(end_seqs_soft: torch.Tensor,
                            end_seqs_hard: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
    """
        This function is implemented to generate sequential-wise masks during training. The idea is to add new negative inf
    values to the existing casual masks.
    Assuming that we already have the following masks:
      0  -inf -inf -inf -inf -inf -inf
      0    0  -inf -inf -inf -inf -inf
      0    0    0  -inf -inf -inf -inf
      0    0    0    0  -inf -inf -inf
      0    0    0    0    0  -inf -inf
      0    0    0    0    0    0  -inf
      0    0    0    0    0    0    0
    Assuming that this input sequence contains several sub-series that starts from 3 and 6. Then we only preserve the
    inner-attention within each series while each the items within the other series will only have access to the last
    element.
    have the following masks:
      0  -inf -inf -inf -inf -inf -inf
      0    0  -inf -inf -inf -inf -inf
      0    0    0  -inf -inf -inf -inf
    -inf -inf   0    0  -inf -inf -inf
    -inf -inf   0    0    0  -inf -inf
    -inf -inf   0  -inf   0   0  -inf
    -inf -inf   0  -inf   0   0    0

    where the start_new_seq indices should be:
      F    F    T    F    T    T    T

    """
    idx_sub_seq, is_last_block = mark_sub_sequences(end_seqs_hard, preseverse_last_block=True)
    # the last triangle needs to be recovered
    end_seqs_soft_ = end_seqs_soft[...,:-1]
    end_seqs_soft[..., :-1] = torch.lerp(
        end_seqs_soft_, 1 - end_seqs_soft_, is_last_block[...,:-1].to(end_seqs_soft_)
    )
    end_seqs_soft = end_seqs_soft.log()

    end_seqs_mask_hard = generate_seq_mask(idx_sub_seq, start_idx=start_idx, )

    mask_soft_log = torch.gather(end_seqs_soft, -1, idx_sub_seq).unsqueeze(-2)
    # https://discuss.pytorch.org/t/torch-where-is-too-slow/100915/4
    mask_soft = torch.lerp(
        end_seqs_soft.unsqueeze(-2) * ~end_seqs_mask_hard,
        mask_soft_log,
        end_seqs_mask_hard.to(mask_soft_log)
    )
    mask_soft = mask_soft * (1-torch.eye(mask_soft.shape[-1], device=mask_soft.device))

    msk_casual = torch.full_like(mask_soft, fill_value=-torch.inf).triu(diagonal=1)
    return mask_soft + msk_casual


def generate_hard_seq_masks(end_seqs_hard: torch.Tensor, start_idx: int = 0, valid_size: torch.Tensor = None) -> torch.Tensor:
    """
    This function is similar to generate_soft_seq_masks, However, it only generates the bool masks values
    """
    idx_sub_seq = mark_sub_sequences(end_seqs_hard)
    if valid_size is not None:
        idx_sub_seq += (end_seqs_hard.shape[-1] - valid_size.to(end_seqs_hard.device))
    end_seqs_mask_hard = generate_seq_mask(idx_sub_seq, start_idx=start_idx)
    return end_seqs_mask_hard
