from dataclasses import dataclass
from transformers.configuration_utils import PretrainedConfig
import inspect


@dataclass
class TokenizerArgs:
    model_name: str
    model_dir: str | None = None
    cache_dir: str | None = None,


@dataclass
class CacheArgs:
    n_kv_heads: int = 8
    n_msks: int = 8

    @classmethod
    def from_dict(cls, env):
        return cls(**{
            k: v for k, v in env.items()
            if k in inspect.signature(cls).parameters
        })

    def __post_init__(self):
        self.n_rep_msk = self.n_kv_heads // self.n_msks


@dataclass
class TransformerArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_msks: int = 1
    n_kv_heads: int | None = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_hidden_dims: int | None = None,
    ffn_dim_multiplier: int | None = 4
    norm_eps: float = 1e-5

    # pos_embedding
    rope_theta: float = 500000
    rope_factor: float = 8.0
    high_freq_factor: float = 4.0
    low_freq_factor: float = 1.0
    max_position_embeddings: int = 8192
    rope_type: str = 'llama3'

    max_batch_size: int = 32
    max_seq_len: int = 2048
    max_seq_len_local: int = 1024

    model_type: str = 'gpt'
    dropout: float = 0.0
    use_flash: bool = True
    ffn_bias: bool = False
    ln_bias: bool = False

    nats_enable: bool = False
    sparse_regularized_value: float = 1e-8
    chunk_size: int = 1
    chunk_merge_method: str = 'mean'
    local_seq_max_length: int = 16
    compress_on_q: bool = False

    apply_pos_emb: bool = True
    has_output_norm: bool = True

    proj_layer_is_ssm: bool = False

    on_ddp: bool = False

    @classmethod
    def from_dict(cls, env):
        return cls(**{
            k: v for k, v in env.items()
            if k in inspect.signature(cls).parameters
        })

    def __post_init__(self):
        if self.model_type == 'gpt':
            pass
        elif self.model_type == 'llama3':
            pass
        else:
            raise NotImplemented(f'Unknown model type: {self.model_type}')

        self.hidden_size = self.dim
        self.num_attention_heads = self.n_heads

        self.rope_scaling = {
            'factor': self.rope_factor,
            'high_freq_factor': self.high_freq_factor,
            'low_freq_factor': self.low_freq_factor,
            'original_max_position_embeddings': self.max_position_embeddings,
            'rope_type': self.rope_type
        }
