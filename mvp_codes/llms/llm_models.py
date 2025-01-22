
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from .llm_utils import Llama32Params



def precompute_rope(self, context_length, att_dim_out, theta_base):
    '''
    Positional encoding is essentially conveying the relevance of key(k) to query(q)
    Need a metric such that it represents the relative position of the key(k) to the query(q)
    but should not modify the vectors of the key(k) and query(q)
    One such metric is the angle between the key(k) and query(q)
    if lets say k@position p1 -> kp1 is k*Rp1 (Rp1 is rotation matrix * p1)
    and q@position p2 -> qp2 is q*Rp2
    now insted of q*k we can use qp2*kp1 --> q*(Rp2*Rp1)*k --> q*R(p2-p1)*k
    R(p2-p1) is the rotation matrix that represents the angle between qp2 and kp1
    R should be a att_dim_out*att_dim_out matrix tob e used with the key and query
    but att_dim_out dimensional rotation matrix is complex, instead authors simplified it by
    using (att_dim_out/2) number of 2D rotation matrices
    R can be precomputed to represent unique rotation
    All 2D rotation matrices can be represented by thetas --> 10000^(-2i/d)
    d = att_dim_out//2, i: 0 to d-1 

    Rotary positional encoder
    theta = 10000^(-2i/d)
    d = att_dim_out
    '''
    
    self.rope_dim = att_dim_out//2
    _j = torch.arange(self.rope_dim)[:self.rope_dim].float()
    thetas = 1.0/(theta_base**(2*_j/self.rope_dim))
    postions = torch.arange(context_length).float()
    angles = postions[:,None]*thetas[None,:] # [CD, AOD//2]
    # check below explanation for concatenation
    angles = torch.cat([angles, angles], dim=-1) # [CD, AOD]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin



def compute_rope(qk, cos, sin):
    '''
    qk: (q or k) [BD, H, CD, AOD]
    [BD, H, CD, AOD]*[AOD//2, AOD//2] (wrong) --> to make it work
    divide qk into two parts qkr and qki (assume real and imaginary)
    [BD, H, CD, AOD//2, 1]->(qkr) and [BD, H, CD, AOD//2, 1]->(qki)
    qk = [qkr;qki] --> [BD, H, CD, AOD//2, 2]
    R2d = [cos1, -sin1; sin1, cos1] --> [CD, 2, 2] such rope_dim 2d matrices
    R*qk = [cos*qkr - sin*qki; sin*qkr + cos*qki]
    this can be represented as [cos*qkr;cos*qki] + [sin*(-qki);sin*qkr] ==
    [cos;cos]*[qkr;qki] + [sin;sin]*[-qki;qkr]
    cos_stacked*qk + sin_stacked*qk_2 (qk_2 = [-qki;qkr])
    
    '''
    BD, H, CD, AOD = qk.shape
    qkr = qk[..., :AOD//2]
    qki = qk[..., AOD//2:]
    qk_2 = torch.cat([-qki, qkr], dim=-1) # [BD, H, CD, AOD]
    cos = cos[None,None,:CD,:]
    cos = sin[None,None,:CD,:] # [1, 1, CD, AOD//2]
    return qk*cos + qk_2*sin
        

class CausalSelfAttentionSH(nn.Module):
    '''
    Single Head Self Attention
    x --> [BD, CD, ED]
    q = W_q(x) == x*w_q --> [BD, CD, ED]*[AID, AODq] --> [BD, CD, AODq]
    (ED = AID, AOD = ED/H)
    k = W_k(x) == x*w_k --> [BD, CD, ED]*[AID, AODk] --> [BD, CD, AODk]
    v = W_v(x) == x*w_v --> [BD, CD, ED]*[AID, AODv] --> [BD, CD, AODv]

    A = softmax(q*k^T/sqrt(AOD)) --> [BD, CD, AODq]*[BD, AODk, CD] --> [BD, CD, CD]
    Mask = torch.triu(torch.ones(1, 1, CD, CD), diagonal=1)
    Mask.masked_fill_(Mask.bool(), -torch.inf) # filling upper triangle above diaognal with -inf, so that softmax will ignore those values
    A = A*Mask
    y = A*v --> [BD, CD, CD]*[BD, CD, AODv] --> [BD, CD, AODv]

    AODq=AODk=AODv=ED/H, AODq,AODk should always be equal. AODv can be different.
    CD also can vary per example.
    '''
    def __init__(self, att_dim_in, att_dim_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.q_proj = nn.Linear(att_dim_in, att_dim_out, bias=qkv_bias)
        self.k_proj = nn.Linear(att_dim_in, att_dim_out, bias=qkv_bias)
        self.v_proj = nn.Linear(att_dim_in, att_dim_out, bias=qkv_bias)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bd, new_cd, ed = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        a = torch.matmul(q, k.transpose(1, 2))
        a.masked_fill_(self.mask.bool()[:new_cd, :new_cd], -torch.inf)
        _, _, AOD = q.shape
        a = torch.softmax(a / (AOD ** 0.5), dim=-1)
        a = self.dropout(a)
        y = torch.matmul(a, v)
        return y
        

class CausalMultiHeadAttSequence(nn.Module):
    '''
    Multi Head Self Attention with single head self attention
    x --> [BD, CD, ED]
    single_head --> [BD, CD, AOD]
    heads --> [BD, CD, AOD*H]

    '''
    def __init__(self, att_dim_in, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        if att_dim_in % num_heads != 0:
            raise ValueError("att_dim_in must be divisible by num_heads.")
        att_dim_out = att_dim_in // num_heads
        self.heads = nn.ModuleList([
            CausalSelfAttentionSH(att_dim_in, att_dim_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(num_heads*att_dim_out, att_dim_in)

    def forward(self, x):
        bd, cd, ed = x.shape
        y = torch.cat([head(x) for head in self.heads], dim=-1)
        y = self.out_proj(y)
        return y


class CausalMultiHeadAttention(nn.Module):
    '''
    Multi Head Self Attention w/o head sequence
    x --> [BD, CD, ED]
    num_heads = H
    q = W_q(x) == x*w_q --> [BD, CD, ED]*[AID, AOD*H] --> [BD, CD, AOD*H] --> [BD, CD, H, AOD] --> [BD, H, CD, AOD]
    (ED = AID, AOD = ED/H)
    k = W_k(x) == x*w_k --> [BD, CD, ED]*[AID, AOD*H] --> [BD, CD, AOD*H] --> [BD, CD, H, AOD] --> [BD, H, CD, AOD]
    v = W_v(x) == x*w_v --> [BD, CD, ED]*[AID, AOD*H] --> [BD, CD, AOD*H] --> [BD, CD, H, AOD] --> [BD, H, CD, AOD]
    out_proj = W_o(y) == y*w_o --> [BD, CD, AOD*H]*[AOD*H, AOD*H] --> [BD, CD, AOD*H]

    replicating num of heads attn 
    A = softmax(q*k^T/sqrt(AOD)) --> [BD, H, CD, AOD]*[BD, H, CD, AOD]^T  --> [BD, H, CD, CD]
    Mask = torch.triu(torch.ones(1, 1, CD, CD), diagonal=1)
    Mask.masked_fill_(Mask.bool(), -torch.inf) # filling upper triangle above diaognal with -inf, so that softmax will ignore those values
    A = A*Mask
    y = A*v --> [BD, H, CD, CD]*[BD, H, CD, AOD] --> [BD, H, CD, AOD] -> [BD, CD, H, AOD] -> [BD, CD, AOD*H]
    y = out_proj(y) --> [BD, CD, AOD*H]*[AOD*H, AOD*H] --> [BD, CD, AOD*H]
    AOD*H should be equal to ED
    '''
    def __init__(self, att_dim_in, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        if att_dim_in % num_heads != 0:
            raise ValueError("att_dim_in must be divisible by num_heads.")
        att_dim_out = att_dim_in // num_heads
        self.q_proj = nn.Linear(att_dim_in, att_dim_out*num_heads, bias=qkv_bias)
        self.k_proj = nn.Linear(att_dim_in, att_dim_out*num_heads, bias=qkv_bias)
        self.v_proj = nn.Linear(att_dim_in, att_dim_out*num_heads, bias=qkv_bias)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(att_dim_out*num_heads, att_dim_out*num_heads)
        self.num_heads = num_heads

    def forward(self, x):
        bd, cd, ed = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        att_dim_out = q.shape[-1] // self.num_heads
        # [BD, CD, H*AOD] --> [BD, CD, H, AOD] --> [BD, H, CD, AOD]
        q = q.view(bd, cd, self.num_heads, att_dim_out).transpose(1, 2)
        k = k.view(bd, cd, self.num_heads, att_dim_out).transpose(1, 2)
        v = v.view(bd, cd, self.num_heads, att_dim_out).transpose(1, 2)
        a = torch.matmul(q, k.transpose(-2, -1)) # [BD, H, CD, CD]
        a.masked_fill_(self.mask.bool()[:cd, :cd], -torch.inf)
        a = torch.softmax(a / (att_dim_out ** 0.5), dim=-1)
        a = self.dropout(a)
        y = torch.matmul(a, v)
        y = y.transpose(1, 2).contiguous().view(bd, cd, att_dim_out*self.num_heads)
        y = self.out_proj(y)
        return y


class CausalMultiHeadAttentionRoPE(nn.Module):
    '''
    Multi Head Self Attention w/o head sequence
    x --> [BD, CD, ED]
    num_heads = H
    q = W_q(x) == x*w_q --> [BD, CD, ED]*[AID, AOD*H] --> [BD, CD, AOD*H] --> [BD, CD, H, AOD] --> [BD, H, CD, AOD]
    (ED = AID, AOD = ED/H)
    k = W_k(x) == x*w_k --> [BD, CD, ED]*[AID, AOD*H] --> [BD, CD, AOD*H] --> [BD, CD, H, AOD] --> [BD, H, CD, AOD]
    v = W_v(x) == x*w_v --> [BD, CD, ED]*[AID, AOD*H] --> [BD, CD, AOD*H] --> [BD, CD, H, AOD] --> [BD, H, CD, AOD]
    out_proj = W_o(y) == y*w_o --> [BD, CD, AOD*H]*[AOD*H, AOD*H] --> [BD, CD, AOD*H]

    replicating num of heads attn 
    A = softmax(q*k^T/sqrt(AOD)) --> [BD, H, CD, AOD]*[BD, H, CD, AOD]^T  --> [BD, H, CD, CD]
    Mask = torch.triu(torch.ones(1, 1, CD, CD), diagonal=1)
    Mask.masked_fill_(Mask.bool(), -torch.inf) # filling upper triangle above diaognal with -inf, so that softmax will ignore those values
    A = A*Mask
    y = A*v --> [BD, H, CD, CD]*[BD, H, CD, AOD] --> [BD, H, CD, AOD] -> [BD, CD, H, AOD] -> [BD, CD, AOD*H]
    y = out_proj(y) --> [BD, CD, AOD*H]*[AOD*H, AOD*H] --> [BD, CD, AOD*H]
    AOD*H should be equal to ED
    '''
    def __init__(self, att_dim_in, context_length, num_heads, dropout, qkv_bias=False, theta_base=10000):
        super().__init__()
        if att_dim_in % num_heads != 0:
            raise ValueError("att_dim_in must be divisible by num_heads.")
        att_dim_out = att_dim_in // num_heads
        self.q_proj = nn.Linear(att_dim_in, att_dim_out*num_heads, bias=qkv_bias)
        self.k_proj = nn.Linear(att_dim_in, att_dim_out*num_heads, bias=qkv_bias)
        self.v_proj = nn.Linear(att_dim_in, att_dim_out*num_heads, bias=qkv_bias)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(att_dim_out*num_heads, att_dim_out*num_heads)
        self.num_heads = num_heads
        cos, sin = precompute_rope(context_length, att_dim_out, theta_base)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def forward(self, x):
        bd, cd, ed = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        att_dim_out = q.shape[-1] // self.num_heads
        # [BD, CD, H*AOD] --> [BD, CD, H, AOD] --> [BD, H, CD, AOD]
        q = q.view(bd, cd, self.num_heads, att_dim_out).transpose(1, 2)
        k = k.view(bd, cd, self.num_heads, att_dim_out).transpose(1, 2)
        v = v.view(bd, cd, self.num_heads, att_dim_out).transpose(1, 2)
        q = compute_rope(q, self.cos, self.sin)
        k = compute_rope(k, self.cos, self.sin)
        a = torch.matmul(q, k.transpose(-2, -1)) # [BD, H, CD, CD]
        a.masked_fill_(self.mask.bool()[:cd, :cd], -torch.inf)
        a = torch.softmax(a / (att_dim_out ** 0.5), dim=-1)
        a = self.dropout(a)
        y = torch.matmul(a, v)
        y = y.transpose(1, 2).contiguous().view(bd, cd, att_dim_out*self.num_heads)
        y = self.out_proj(y)
        return y




class CausalMultiHeadFlashAttention(nn.Module):
    '''
    Multi Head Self Attention w/o head sequence
    x --> [BD, CD, ED]
    num_heads = H
    q = W_q(x) == x*w_q --> [BD, CD, ED]*[AID, AOD*H] --> [BD, CD, AOD*H] --> [BD, CD, H, AOD] --> [BD, H, CD, AOD]
    (ED = AID, AOD = ED/H)
    k = W_k(x) == x*w_k --> [BD, CD, ED]*[AID, AOD*H] --> [BD, CD, AOD*H] --> [BD, CD, H, AOD] --> [BD, H, CD, AOD]
    v = W_v(x) == x*w_v --> [BD, CD, ED]*[AID, AOD*H] --> [BD, CD, AOD*H] --> [BD, CD, H, AOD] --> [BD, H, CD, AOD]
    out_proj = W_o(y) == y*w_o --> [BD, CD, AOD*H]*[AOD*H, AOD*H] --> [BD, CD, AOD*H]

    replicating num of heads attn 
    A = softmax(q*k^T/sqrt(AOD)) --> [BD, H, CD, AOD]*[BD, H, CD, AOD]^T  --> [BD, H, CD, CD]
    Mask = torch.triu(torch.ones(1, 1, CD, CD), diagonal=1)
    Mask.masked_fill_(Mask.bool(), -torch.inf) # filling upper triangle above diaognal with -inf, so that softmax will ignore those values
    A = A*Mask
    y = A*v --> [BD, H, CD, CD]*[BD, H, CD, AOD] --> [BD, H, CD, AOD] -> [BD, CD, H, AOD] -> [BD, CD, AOD*H]
    y = out_proj(y) --> [BD, CD, AOD*H]*[AOD*H, AOD*H] --> [BD, CD, AOD*H]
    AOD*H should be equal to ED
    '''
    def __init__(self, att_dim_in, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        if att_dim_in % num_heads != 0:
            raise ValueError("att_dim_in must be divisible by num_heads.")
        att_dim_out = att_dim_in // num_heads
        self.q_proj = nn.Linear(att_dim_in, att_dim_out*num_heads, bias=qkv_bias)
        self.k_proj = nn.Linear(att_dim_in, att_dim_out*num_heads, bias=qkv_bias)
        self.v_proj = nn.Linear(att_dim_in, att_dim_out*num_heads, bias=qkv_bias)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(att_dim_out*num_heads, att_dim_out*num_heads)
        self.num_heads = num_heads
        cos, sin = precompute_rope(context_length, att_dim_out, 10000)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def forward(self, x):
        bd, cd, ed = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        att_dim_out = q.shape[-1] // self.num_heads
        # [BD, CD, H*AOD] --> [BD, CD, H, AOD] --> [BD, H, CD, AOD]
        q = q.view(bd, cd, self.num_heads, att_dim_out).transpose(1, 2)
        k = k.view(bd, cd, self.num_heads, att_dim_out).transpose(1, 2)
        v = v.view(bd, cd, self.num_heads, att_dim_out).transpose(1, 2)
        use_dropout = 0 if self.training else self.dropout
        # if is_causal: False, have to provide mask
        q = compute_rope(q, self.cos, self.sin)
        k = compute_rope(k, self.cos, self.sin)
        y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout=use_dropout, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bd, cd, att_dim_out*self.num_heads)
        y = self.out_proj(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, att_dim_in, ffn_dim, dtype):
        super().__init__()
        self.fc1 = nn.Linear(att_dim_in, ffn_dim, dtype=dtype, bias=False)
        self.fc2 = nn.Linear(att_dim_in, ffn_dim, dtype=dtype, bias=False)
        self.fc3 = nn.Linear(ffn_dim, att_dim_in, dtype=dtype, bias=False)

    def forward(self, x):
        return self.fc3(nn.functional.silu(self.fc1(x)) * self.fc2(x))

class TransformerBlock(nn.Module):
    def __init__(self, att_dim_in, context_length, num_heads, ffn_dim, dropout, dtype=torch.float32, qkv_bias=False):
        super().__init__()
        self.attention = CausalMultiHeadAttentionRoPE(att_dim_in, context_length, num_heads, dropout, qkv_bias)
        self.ff = FeedForward(att_dim_in, ffn_dim, dtype)
        # self.ff = nn.Sequential(
        #     nn.Linear(att_dim_in, ffn_dim, dtype=dtype, bias=False),
        #     nn.GELU(),
        #     nn.Linear(ffn_dim, att_dim_in, dtype=dtype, bias=False)
        # )
        self.norm1 = nn.RMSNorm(att_dim_in, eps=1e-6)
        self.norm2 = nn.RMSNorm(att_dim_in, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # y = self.attention(self.norm1(x))
        # x = x + self.dropout(y)
        x = x + self.attention(self.norm1(x))
        # y = self.ff(self.norm2(x))
        # x = x + self.dropout(y)
        x = x + self.ff(self.norm2(x))
        return x

class LlmModel(nn.Module):
    def __init__(self, params: Llama32Params):
        super().__init__()
        self.params = params
        self.tok_embed = nn.Embedding(params.vocab_size, params.embed_dim)
        self.pos_embed = nn.Embedding(params.cntx_len, params.embed_dim)
        self.dropout = nn.Dropout(params.dropout)
        self.trf_blocks = nn.Sequential(*[
            TransformerBlock(params.embed_dim, params.cntx_len, params.num_heads, params.ffn_dim, params.dropout)
            for _ in range(params.num_layers)
        ])
        self.final_norm = nn.RMSNorm(params.embed_dim, eps=1e-6)
        self.out_head = nn.Linear(params.embed_dim, params.vocab_size, bias=False)
        # Tie weights
        self.out_head.weight = self.tok_embed.weight

    def forward(self, tokens):
        bd, cd = tokens.shape
        x = self.tok_embed(tokens) + self.pos_emb(torch.arange(cd, device=tokens.device))
        x = self.dropout(x)
        x = self.trf_blocks(x, x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
