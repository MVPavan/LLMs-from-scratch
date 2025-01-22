from pathlib import Path
from pydantic import BaseModel
from omegaconf import OmegaConf
llama3_config = OmegaConf.load('mvp_codes/llama3/llama3.yml')

'''
BD --> Batch size
CD --> Context length
ED --> Embedding dimension
H --> Number of heads
AID = ED --> Attention in dimension
AOD = AID/H = ED/H --> Attention out dimension
L --> Number of layers
FFD --> Feedforward dimension
'''
class Llama32Params(BaseModel):
    vocab_size: int = 128256 # 
    cntx_len: int = 131072
    embed_dim: int = 2048
    num_heads: int = 32
    num_layers: int = 16
    ffn_dim: int = 8192
    dropout: float = 0.1

    def __init__(self):
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads.")
        self.att_dim_in = self.embed_dim
        self.att_dim_out = self.embed_dim//self.num_heads