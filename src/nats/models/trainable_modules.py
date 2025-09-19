from nats.models.transformer.components import RMSNorm as llama_rmsnorm,  LayerNorm as gpt_ln

from torch.nn import LayerNorm as torch_ln, Conv1d
from torch.nn import Linear, Embedding

TRANSFORMER_TRAINABLE_MODULES = {llama_rmsnorm, gpt_ln, torch_ln, Linear, Embedding}
