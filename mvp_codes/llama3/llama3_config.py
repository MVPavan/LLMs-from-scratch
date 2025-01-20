from pathlib import Path
from pydantic import BaseModel
from omegaconf import OmegaConf
llama3_config = OmegaConf.load('mvp_codes/llama3/llama3_config.yaml')


class Llama32Params(BaseModel):
    vocab_size: int = 128256
    cntx_len: int = 131072
    embed_dim: int = 2048
    num_heads: int = 32
    num_layers: int = 16
    ffn_dim: int = 8192