from typing import Optional
from pydantic import BaseModel, Field, model_validator
from omegaconf import OmegaConf, DictConfig
import torch
# Load YAML into OmegaConf
config_path = "mvp_codes/llms/llm_configs.yml"
llm_config = OmegaConf.load(config_path)

'''
BD --> Batch size
CD --> Context length
ED --> Embedding dimension
NH --> Number of heads
AD = ED --> Attention in dimension
HD = AD/NH = ED/NH --> Attention out dimension (for single head)
NL --> Number of layers
FFD --> Feedforward dimension
'''

def rescale_theta(theta_old, context_length_old, context_length_new):
    scaling_factor = context_length_new / context_length_old
    theta_new = theta_old * scaling_factor
    return int(theta_new)

class LLMParams(BaseModel):
    vocab_size: int = 128256
    context_length: int = 131072
    embed_dim: int = 2048
    num_heads: int = 32
    num_kv_groups: int = 8
    num_layers: int = 16
    ffn_dim: int = 8192
    dropout: float = 0
    rope_base: int = 500000
    qkv_bias: bool = False
    dtype: torch.dtype = torch.float32
    att_dim: Optional[int] = Field(default=None) # "Attention dimension, computed post-init"
    head_dim: Optional[int] = Field(default=None) # "Head dimension, computed post-init"
    class Config:
        arbitrary_types_allowed = True 
        
    @model_validator(mode='after')
    def validate_parameters(cls, values):
        embed_dim = values.embed_dim
        num_heads = values.num_heads
        num_kv_groups = values.num_kv_groups

        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads.")

        if num_kv_groups > 0 and num_heads % num_kv_groups != 0:
            raise ValueError("Number of key-value groups must be divisible by number of heads.")

        if values.att_dim is not None and values.att_dim != embed_dim:
            raise ValueError("Attention dimension must match embedding dimension.")
        else:
            values.att_dim = embed_dim

        if values.head_dim is not None and values.head_dim != embed_dim // num_heads:
            raise ValueError("Head dimension must match embedding dimension divided by number of heads.")
        else:
            values.head_dim = embed_dim // num_heads
        # 131072 ctx of llama32
        if values.rope_base == 0:
            values.rope_base = 500000
        values.rope_base = rescale_theta(values.rope_base, 131072, values.context_length)
        return values

def get_llm_params(config: DictConfig):
    config_dict = OmegaConf.to_container(config, resolve=True)
    validated_params = LLMParams(**config_dict)
    return validated_params
