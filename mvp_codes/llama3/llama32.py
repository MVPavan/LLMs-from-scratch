
import torch
import torch.nn as nn
from torch.utils.data import Dataset



class Llama32Dataset(Dataset):
    def __init__(self, params: Llama32Params, data):
        self.params = params
        self.data = data

    def __len__(self):
        return self.params.cntx_len

    def __getitem__(self, idx):
        return torch.randint(0, self.params.vocab_size, (self.params.cntx_len,))

class Llama32(nn.Module):
    def __init__(self, params: Llama32Params):
        super().__init__()
        self.params = params
        self.tokenizer = Tokenizer(params.vocab_size)
        self.embed = nn.Embedding(params.vocab_size, params.embed_dim)
        self.transformer = nn.Transformer(
            d_model=params.embed_dim,
            nhead=params.num_heads,
            dim_feedforward=params.ffn_dim,
        )
        self.proj = nn.Linear(params.embed_dim, params.vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x, x)
        x = self.proj(x)
        return x
