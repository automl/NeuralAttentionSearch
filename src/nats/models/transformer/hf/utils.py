import torch


@torch.compile
def prepare_qkv_for_flash_infer(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                bsz: int, nh_kv: int, num_kv_groups: int, q_len: int, kv_len: int, head_dim: int):
    """
    Prepare QKV to fit for the flashinfer block sparse attention
    Args:
        q: torch.Tensor of shape [bsz, nh_qo, q_len, head_dim]
        k: torch.Tensor of shape [bsz, nh_kv, kv_len, head_dim]
        v: torch.Tensor of shape [bsz, nh_kv, kv_len, head_dim]
        bsz: int, batch size
        nh_qo: int, number of qo heads
        nh_kv: int, number of kv heads
        q_len: int, q length
        kv_len: int, kv length,
        head_dim: int head dims

    Returns:

    """
    q = q.view(bsz * nh_kv, num_kv_groups * q_len, head_dim)
    k = k.view(bsz * kv_len * nh_kv, 1, head_dim)
    v = v.view(bsz * kv_len * nh_kv, 1, head_dim)
    return q, k, v


@torch.compile
def rescale_sparse_attn(o: torch.Tensor, lse: torch.Tensor, n_invalid: torch.Tensor,
                        q_len: int,
                        bsz: int,
                        num_qo_heads: int,
                        head_dim: int):
    o_scale = torch.exp2(lse) / (torch.exp2(lse) - n_invalid)
    o = o * o_scale.unsqueeze(-1)
    return o.view(bsz, num_qo_heads, q_len, head_dim)
