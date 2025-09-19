from typing import Optional, Tuple
import copy
import torch

from transformers import Cache
from transformers.models.mistral import MistralForCausalLM
from transformers.models.mistral.modeling_mistral import MistralAttention, logger, apply_rotary_pos_emb, repeat_kv
from torch import nn
from torch.nn import functional as F
import types
import math
from torchtune.modules.peft.lora import LoRALinear
from nats.models.model_configuration import TransformerArgs
from nats.models.transformer.triton.nats_chunk_attention import nats_chunk_attention
from nats.models.transformer.triton.nats_flashattn_fwd import nats_prefill
from nats.models.transformer.utils import repeat_masks
from nats.chunk_utils import max_pooling_seqlen, average_pooling_seqlen, ChunkMergeType


def forward_mistral_nats_chunk_two_way(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if not self.training:
        # full evaluation
        full_query_states = self.q_proj(hidden_states)
        full_key_states = self.k_proj(hidden_states)
        full_value_states = self.v_proj(hidden_states)
        full_query_states = full_query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        full_key_states = full_key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        full_value_states = full_value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        cos, sin = self.rotary_emb(full_value_states, position_ids)
        full_query_states, full_key_states = apply_rotary_pos_emb(full_query_states, full_key_states, cos, sin)

        full_key_states = repeat_kv(full_key_states, self.num_key_value_groups)
        full_value_states = repeat_kv(full_value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : full_key_states.shape[-2]]

        if full_query_states.device.type == "cuda" and causal_mask is not None:
            full_query_states = full_query_states.contiguous()
            full_key_states = full_key_states.contiguous()
            full_value_states = full_value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        full_attn_output = torch.nn.functional.scaled_dot_product_attention(
            full_query_states.contiguous(),
            full_key_states.contiguous(),
            full_value_states.contiguous(),
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        full_attn_output = full_attn_output.transpose(1, 2).contiguous()
        full_attn_output = full_attn_output.view(bsz, q_len, -1)
        full_attn_output = self.o_proj(full_attn_output)
        return full_attn_output, None, past_key_value
    else:
        nats_query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1,
                                                                                                                 2)
        nats_key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads,
                                                          self.head_dim).transpose(1, 2)
        nats_value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads,
                                                            self.head_dim).transpose(1, 2)

        # Now we want to insert the proj layer and compute the token infor states
        x_len = hidden_states.shape[1]
        if self.chunk_size > 1:
            pad_size = math.ceil(x_len / self.chunk_size) * self.chunk_size - x_len
            x_subsample = self.chunk_merge_func(hidden_states, self.chunk_size, pad_size, )
            logit = self.proj_layer(x_subsample.detach()).unflatten(-1, (self.num_key_value_heads, 3))
        else:
            logit = self.proj_layer(hidden_states.detach()).unflatten(-1, (self.num_key_value_heads, 3))
        token_states_info = nn.functional.gumbel_softmax(logit, tau=1., hard=True, dim=-1)  # (B, N_CTX, H, N_OPTs)

        # the first and the last elements are assigned to the first element
        self.one_hot_values = self.one_hot_values.to(token_states_info)
        token_states_info[:, [0, -1], :] = self.one_hot_values
        token_states_info = torch.transpose(token_states_info, 1, 2).contiguous()  # (B, H, N_CTX, N_OPTS)

        N_CTX = token_states_info.shape[2]
        end_seq_idx = N_CTX - 1 - torch.flip(torch.cummax(torch.flip(token_states_info[..., 0], (-1,)), -1)[1], (-1,))

        n_valid_tokens = torch.where(
            (token_states_info[..., 0] == 1.) | (end_seq_idx == N_CTX - 1) & (token_states_info[..., 2] != 1.), 1, 0
        ).float()

        self.sparse_size = torch.mean(n_valid_tokens).detach().cpu()

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(nats_value_states, position_ids)
        else:
            cos, sin = position_embeddings
        nats_query_states, nats_key_states = apply_rotary_pos_emb(nats_query_states, nats_key_states, cos, sin)

        sm_scale = 1 / math.sqrt(self.head_dim)
        nats_attn_output = nats_chunk_attention(
            nats_query_states.contiguous(), nats_key_states.contiguous(), nats_value_states.contiguous(),
            token_states_info,
            end_seq_idx,
            nats_chunk_size=self.chunk_size,
            causal=True, sm_scale=sm_scale,
            n_rep=self.num_key_value_groups,
            sparse_regularized_value=self.sparse_regularized_value,
            local_seq_max_length=self.local_seq_max_length,
        )
        nats_attn_output = nats_attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        nats_attn_output = self.o_proj(nats_attn_output)
        return nats_attn_output, None, past_key_value


def enable_mistral_nats_chunk_training(
    model: MistralForCausalLM,
    transformer_args: TransformerArgs,
    n_options: int = 3
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    one_hot_values = torch.zeros(n_options, device=device, dtype=dtype)
    one_hot_values[0] = 1.

    # we first disable all the gradient for the raw model
    for n, p in model.named_parameters():
        p.requires_grad = False

    for layer in model.model.layers:
        module: MistralAttention = layer.self_attn
        module.proj_layer = nn.Linear(
            module.hidden_size, module.num_key_value_heads * n_options, device=device, dtype=dtype,
            bias=False
        )
        setattr(module, 'n_reps', module.num_heads // transformer_args.n_msks)
        setattr(module, 'sparse_regularized_value', transformer_args.sparse_regularized_value)
        setattr(module, 'local_seq_max_length', transformer_args.local_seq_max_length)
        setattr(module, 'sparse_size', 0.)
        setattr(module, 'n_options', n_options)
        setattr(module, 'chunk_size', transformer_args.chunk_size)
        if transformer_args.chunk_merge_method == 'mean':
            module.chunk_merge_func = average_pooling_seqlen
        elif transformer_args.chunk_merge_method == 'max':
            module.chunk_merge_func = max_pooling_seqlen
        else:
            raise NotImplementedError(f"Unknown merge method: {module.chunk_merge_func}")
        setattr(module, 'chunk_merge_method', transformer_args.chunk_merge_method)
        setattr(module, 'one_hot_values', one_hot_values)
        module.forward = types.MethodType(forward_mistral_nats_chunk_two_way, module)


def forward_mistral_nats_chunk_one_way(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    x_len = hidden_states.shape[1]
    pre_fill_chunk_size = past_key_value.get_nats_chunk_fill_size(self.layer_idx)
    is_pre_filling = position_ids[0, 0] == 0

    if pre_fill_chunk_size + x_len < self.chunk_size:
        past_key_value.update_nats_cached_x_in_chunk(hidden_states, self.layer_idx, self.chunk_merge_method)
        token_states_info = self.one_hot_values.to(hidden_states).reshape((1, 1, 1, -1)).expand(hidden_states.shape[0], hidden_states.shape[1], 1, -1)
    else:
        x_subsample, pad_size = past_key_value.update_nats_cached_x_across_chunk(hidden_states,
                                                                        pre_fill_chunk_size,
                                                                        self.layer_idx,
                                                                        self.chunk_merge_method)

        logit = self.proj_layer(x_subsample).unflatten(-1, (self.num_key_value_heads, 3))

        # in this case, there is no need to do gumble softmax
        index = logit.max(-1, keepdim=True)[1]
        if position_ids[0, 0] < self.chunk_size:
            index[:, 0] = 0
        if pad_size != 0:
            index[:, -1] = 0
        token_states_info = torch.zeros_like(logit, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        token_states_info = torch.transpose(token_states_info, 1, 2).contiguous()  # (B, H, N_CTX, N_OPTS)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, 'token_states_info': token_states_info}
        cache_kwargs = {'token_states_info': token_states_info}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx,
                                                         cache_kwargs=cache_kwargs)

    if is_pre_filling:
        # for prefilling, we use our forward kernel to reduce memory consumption
        #BSZ, NHEADS, N_CTX = query_states.shape[:3]
        end_seqs_info = token_states_info[..., 0].clone().contiguous()
        end_seqs_info = token_states_info[..., 0].clone().contiguous()
        end_seqs_info[..., -1] = 1.
        end_seq_idx = end_seqs_info.shape[-1] - 1 - torch.flip(torch.cummax(torch.flip(end_seqs_info, (-1,)), -1)[1], (-1,))

        attn_output = nats_prefill(query_states.contiguous(), key_states.contiguous(), value_states.contiguous(),
                                   sliding_tokens=token_states_info[..., 2].contiguous(),
                                   end_seqs_idx=end_seq_idx,
                                   sm_scale=1 / math.sqrt(self.head_dim),
                                   n_rep=self.num_key_value_groups,
                                   local_seq_max_length=self.local_seq_max_length,
                                   )

    #if not is_pre_filling:
    else:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        mask = past_key_value.generate_mask(layer_idx=self.layer_idx, x_len=q_len)
        mask = repeat_masks(mask, n_rep=self.num_key_value_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
            )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    if self.executor is None:
        past_key_value.post_update(layer_idx=self.layer_idx, x_len=q_len, )
    else:
        self.post_update_is_done = self.executor.submit(past_key_value.post_update, layer_idx=self.layer_idx,
                                                        x_len=q_len, )

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def enable_mistral_nats_chunk_eval(
    model: MistralForCausalLM,
    transformer_args: TransformerArgs,
    n_options: int = 3
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    one_hot_values = torch.zeros(n_options, device=device, dtype=dtype)
    one_hot_values[0] = 1.

    for layer in model.model.layers:
        module: MistralAttention = layer.self_attn
        module.proj_layer = nn.Linear(
            module.hidden_size, module.num_key_value_heads * n_options, device=device, dtype=dtype,
            bias=False
        )
        setattr(module, 'n_reps', module.num_heads // transformer_args.n_msks)
        setattr(module, 'sparse_regularized_value', transformer_args.sparse_regularized_value)
        setattr(module, 'local_seq_max_length', transformer_args.local_seq_max_length)
        setattr(module, 'sparse_size', 0.)
        setattr(module, 'n_options', n_options)
        setattr(module, 'one_hot_values', one_hot_values)
        module.forward = types.MethodType(forward_mistral_nats_chunk_one_way, module)
