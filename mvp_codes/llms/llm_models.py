import torch
import torch.nn as nn
from .llm_utils import LLMParams

# NOTE: For dimension explanation, refer to the llm_utils.py

class SharedBuffers:
    _buffers = {}
    @staticmethod
    def get_buffers(context_length, head_dim, rope_base):
        '''
        shared tensors in buffers are only created once and share same memory
        '''
        key = (context_length, head_dim, rope_base)
        if key not in SharedBuffers._buffers:
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = precompute_rope(context_length, head_dim, rope_base)
            SharedBuffers._buffers[key] = (mask, cos, sin)
        return SharedBuffers._buffers[key]


def precompute_rope(context_length, head_dim, rope_base):
    '''
    Positional encoding is essentially conveying the relevance of key(k) to query(q)
    Need a metric such that it represents the relative position of the key(k) to the query(q)
    but should not modify the vectors of the key(k) and query(q)
    One such metric is the angle between the key(k) and query(q)
    if lets say k@position p1 -> kp1 is k*Rp1 (Rp1 is rotation matrix * p1)
    and q@position p2 -> qp2 is q*Rp2
    now insted of q*k we can use qp2*kp1 --> q*(Rp2*Rp1)*k --> q*R(p2-p1)*k
    R(p2-p1) is the rotation matrix that represents the angle between qp2 and kp1

    R should be a head_dim*head_dim matrix tob e used with the key and query
    but head_dim dimensional rotation matrix is complex, instead authors simplified it by
    using (head_dim/2) number of 2D rotation matrices
    R can be precomputed to represent unique rotation
    All 2D rotation matrices can be represented by thetas --> 10000^(-2i/d)
    d = head_dim//2, i: 0 to d-1 

    Rotary positional encoder
    theta = 10000^(-2i/d)
    d = head_dim
    '''
    
    rope_dim = head_dim//2
    _j = torch.arange(rope_dim)[:rope_dim].float()
    thetas = 1.0/(rope_base**(2*_j/rope_dim))
    postions = torch.arange(context_length).float()
    angles = postions[:,None]*thetas[None,:] # [CD, HD//2]
    # check below explanation for concatenation
    angles = torch.cat([angles, angles], dim=-1) # [CD, HD]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin



def compute_rope(qk, cos, sin):
    '''
    qk: (q or k) [BD, NH, CD, HD]
    [BD, NH, CD, HD]*[HD//2, HD//2] (matmul incompatible dimensions) --> to make it work
    divide qk into two parts qkr and qki (assume real and imaginary)
    [BD, NH, CD, HD//2, 1]->(qkr) and [BD, NH, CD, HD//2, 1]->(qki)
    qk = [qkr;qki] --> [BD, NH, CD, HD//2, 2]
    R2d = [cos1, -sin1; sin1, cos1] --> [CD, 2, 2] such rope_dim 2d matrices
    R*qk = [cos*qkr - sin*qki; sin*qkr + cos*qki]
    this can be represented as [cos*qkr;cos*qki] + [sin*(-qki);sin*qkr] ==
    [cos;cos]*[qkr;qki] + [sin;sin]*[-qki;qkr]
    cos_stacked*qk + sin_stacked*qk_2 (qk_2 = [-qki;qkr])
    
    '''
    BD, NH, CD, HD = qk.shape
    qkr = qk[..., :HD//2]
    qki = qk[..., HD//2:]
    qk_2 = torch.cat([-qki, qkr], dim=-1) # [BD, NH, CD, HD]
    cos = cos[None,None,:CD,:]
    cos = sin[None,None,:CD,:] # [1, 1, CD, HD//2]
    return qk*cos + qk_2*sin
        

class CausalSelfAttentionSH(nn.Module):
    '''
    Single Head Self Attention
    x --> [BD, CD, ED]
    q = W_q(x) == x*w_q --> [BD, CD, ED]*[AD, HDq] --> [BD, CD, HDq]
    (ED = AD, HD = ED/NH)
    k = W_k(x) == x*w_k --> [BD, CD, ED]*[AD, HDk] --> [BD, CD, HDk]
    v = W_v(x) == x*w_v --> [BD, CD, ED]*[AD, HDv] --> [BD, CD, HDv]

    A = softmax(q*k^T/sqrt(HD)) --> [BD, CD, HDq]*[BD, HDk, CD] --> [BD, CD, CD]
    Mask = torch.triu(torch.ones(1, 1, CD, CD), diagonal=1)
    Mask.masked_fill_(Mask.bool(), -torch.inf) # filling upper triangle above diaognal with -inf, so that softmax will ignore those values
    A = A*Mask
    y = A*v --> [BD, CD, CD]*[BD, CD, HDv] --> [BD, CD, HDv]

    HDq=HDk=HDv=ED/NH, HDq,HDk should always be equal. HDv can be different.
    CD also can vary per input.
    '''
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.q_proj = nn.Linear(params.att_dim, params.head_dim, bias=params.qkv_bias)
        self.k_proj = nn.Linear(params.att_dim, params.head_dim, bias=params.qkv_bias)
        self.v_proj = nn.Linear(params.att_dim, params.head_dim, bias=params.qkv_bias)
        self.register_buffer('mask', torch.triu(torch.ones(params.context_length, params.context_length), diagonal=1))
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        bd, new_cd, ed = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        a = torch.matmul(q, k.transpose(1, 2))
        a.masked_fill_(self.mask.bool()[:new_cd, :new_cd], -torch.inf)
        _, _, HD = q.shape
        a = torch.softmax(a / (HD ** 0.5), dim=-1)
        a = self.dropout(a)
        y = torch.matmul(a, v)
        return y
        

class CausalMultiHeadAttSequence(nn.Module):
    '''
    Multi Head Self Attention with single head self attention
    x --> [BD, CD, ED]
    single_head --> [BD, CD, HD]
    heads --> [BD, CD, HD*NH]

    '''
    def __init__(self, params):
        super().__init__()
        self.params = params
        if params.att_dim % params.num_heads != 0:
            raise ValueError("att_dim must be divisible by num_heads.")
        self.heads = nn.ModuleList([CausalSelfAttentionSH(params) for _ in range(params.num_heads)])
        self.out_proj = nn.Linear(params.num_heads * params.head_dim, params.att_dim)

    def forward(self, x):
        bd, cd, ed = x.shape
        y = torch.cat([head(x) for head in self.heads], dim=-1)
        y = self.out_proj(y)
        return y


class CausalMultiHeadAttention(nn.Module):
    '''
    Multi Head Self Attention w/o head sequence
    x --> [BD, CD, ED]
    num_heads = NH
    q = W_q(x) == x*w_q --> [BD, CD, ED]*[AD, HD*NH] --> [BD, CD, HD*NH] --> [BD, CD, NH, HD] --> [BD, NH, CD, HD]
    (ED = AD, HD = ED/NH)
    k = W_k(x) == x*w_k --> [BD, CD, ED]*[AD, HD*NH] --> [BD, CD, HD*NH] --> [BD, CD, NH, HD] --> [BD, NH, CD, HD]
    v = W_v(x) == x*w_v --> [BD, CD, ED]*[AD, HD*NH] --> [BD, CD, HD*NH] --> [BD, CD, NH, HD] --> [BD, NH, CD, HD]
    out_proj = W_o(y) == y*w_o --> [BD, CD, HD*NH]*[HD*NH, HD*NH] --> [BD, CD, HD*NH]

    replicating num of heads attn 
    A = softmax(q*k^T/sqrt(HD)) --> [BD, NH, CD, HD]*[BD, NH, CD, HD]^T  --> [BD, NH, CD, CD]
    Mask = torch.triu(torch.ones(1, 1, CD, CD), diagonal=1)
    Mask.masked_fill_(Mask.bool(), -torch.inf) # filling upper triangle above diaognal with -inf, so that softmax will ignore those values
    A = A*Mask
    y = A*v --> [BD, NH, CD, CD]*[BD, NH, CD, HD] --> [BD, NH, CD, HD] -> [BD, CD, NH, HD] -> [BD, CD, HD*NH]
    y = out_proj(y) --> [BD, CD, HD*NH]*[HD*NH, HD*NH] --> [BD, CD, HD*NH]
    HD*NH should be equal to ED
    '''
    def __init__(self, params):
        super().__init__()
        self.params = params
        if params.att_dim % params.num_heads != 0:
            raise ValueError("att_dim must be divisible by num_heads.")
        self.q_proj = nn.Linear(params.att_dim, params.head_dim * params.num_heads, bias=params.qkv_bias)
        self.k_proj = nn.Linear(params.att_dim, params.head_dim * params.num_heads, bias=params.qkv_bias)
        self.v_proj = nn.Linear(params.att_dim, params.head_dim * params.num_heads, bias=params.qkv_bias)
        self.register_buffer('mask', torch.triu(torch.ones(params.context_length, params.context_length), diagonal=1))
        self.dropout = nn.Dropout(params.dropout)
        self.out_proj = nn.Linear(params.head_dim * params.num_heads, params.head_dim * params.num_heads)
        self.num_heads = params.num_heads

    def forward(self, x):
        bd, cd, ed = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # [BD, CD, NH*HD] --> [BD, CD, NH, HD] --> [BD, NH, CD, HD]
        q = q.view(bd, cd, self.num_heads, self.params.head_dim).transpose(1, 2)
        k = k.view(bd, cd, self.num_heads, self.params.head_dim).transpose(1, 2)
        v = v.view(bd, cd, self.num_heads, self.params.head_dim).transpose(1, 2)
        a = torch.matmul(q, k.transpose(-2, -1)) # [BD, NH, CD, CD]
        a.masked_fill_(self.mask.bool()[:cd, :cd], -torch.inf)
        a = torch.softmax(a / (self.params.head_dim ** 0.5), dim=-1)
        a = self.dropout(a)
        y = torch.matmul(a, v)
        y = y.transpose(1, 2).contiguous().view(bd, cd, self.params.head_dim*self.num_heads)
        y = self.out_proj(y)
        return y


class CausalMultiHeadAttentionRoPE(nn.Module):
    '''
    Multi Head Self Attention Rope
    x --> [BD, CD, ED]
    num_heads = NH
    q = W_q(x) == x*w_q --> [BD, CD, ED]*[AD, HD*NH] --> [BD, CD, HD*NH] --> [BD, CD, NH, HD] --> [BD, NH, CD, HD]
    (ED = AD, HD = ED/NH)
    k = W_k(x) == x*w_k --> [BD, CD, ED]*[AD, HD*NH] --> [BD, CD, HD*NH] --> [BD, CD, NH, HD] --> [BD, NH, CD, HD]
    v = W_v(x) == x*w_v --> [BD, CD, ED]*[AD, HD*NH] --> [BD, CD, HD*NH] --> [BD, CD, NH, HD]
    out_proj = W_o(y) == y*w_o --> [BD, CD, HD*NH]*[HD*NH, HD*NH] --> [BD, CD, HD*NH]

    replicating num of heads attn 
    A = softmax(q*k^T/sqrt(HD)) --> [BD, NH, CD, HD]*[BD, NH, CD, HD]^T  --> [BD, NH, CD, CD]
    Mask = torch.triu(torch.ones(1, 1, CD, CD), diagonal=1)
    Mask.masked_fill_(Mask.bool(), -torch.inf) # filling upper triangle above diaognal with -inf, so that softmax will ignore those values
    A = A*Mask
    y = A*v --> [BD, NH, CD, CD]*[BD, NH, CD, HD] --> [BD, NH, CD, HD] -> [BD, CD, NH, HD] -> [BD, CD, HD*NH]
    y = out_proj(y) --> [BD, CD, HD*NH]*[HD*NH, HD*NH] --> [BD, CD, HD*NH]
    HD*NH should be equal to ED
    '''
    def __init__(self, params):
        super().__init__()
        self.params = params
        if params.att_dim % params.num_heads != 0:
            raise ValueError("att_dim must be divisible by num_heads.")
        self.q_proj = nn.Linear(params.att_dim, params.head_dim * params.num_heads, bias=params.qkv_bias)
        self.k_proj = nn.Linear(params.att_dim, params.head_dim * params.num_heads, bias=params.qkv_bias)
        self.v_proj = nn.Linear(params.att_dim, params.head_dim * params.num_heads, bias=params.qkv_bias)
        self.register_buffer('mask', torch.triu(torch.ones(params.context_length, params.context_length), diagonal=1))
        self.dropout = nn.Dropout(params.dropout)
        self.out_proj = nn.Linear(params.head_dim * params.num_heads, params.head_dim * params.num_heads)
        self.num_heads = params.num_heads
        cos, sin = precompute_rope(params.context_length, params.head_dim, params.rope_base)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def forward(self, x):
        bd, cd, ed = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # [BD, CD, NH*HD] --> [BD, CD, NH, HD] --> [BD, NH, CD, HD]
        q = q.view(bd, cd, self.num_heads, self.params.head_dim).transpose(1, 2)
        k = k.view(bd, cd, self.num_heads, self.params.head_dim).transpose(1, 2)
        v = v.view(bd, cd, self.num_heads, self.params.head_dim).transpose(1, 2)
        q = compute_rope(q, self.cos, self.sin)
        k = compute_rope(k, self.cos, self.sin)
        a = torch.matmul(q, k.transpose(-2, -1)) # [BD, NH, CD, CD]
        a.masked_fill_(self.mask.bool()[:cd, :cd], -torch.inf)
        a = torch.softmax(a / (self.params.head_dim ** 0.5), dim=-1)
        a = self.dropout(a)
        y = torch.matmul(a, v)
        y = y.transpose(1, 2).contiguous().view(bd, cd, self.params.head_dim*self.num_heads)
        y = self.out_proj(y)
        return y


class CausalMultiHeadGroupAttention(nn.Module):
    '''
    Multi Head Group Query Attention
    In GQA, we reduce the number of key and value projections by sharing them among multiple attention heads
    Each attention head still has its unique query, but these queries attend to the same group of keys and values
    x --> [BD, CD, ED]
    num_heads = NH
    number of kv_groups = Nkv (if Nkv = 1, it is grouped QA, if Nkv = NH, it is full/normal QA)
    q = W_q(x) == x*w_q --> [BD, CD, ED]*[AD, HD*NH] --> [BD, CD, HD*NH] --> [BD, CD, NH, HD] --> [BD, NH, CD, HD]
    (ED = AD, HD = ED/NH)
    For grouped kv instead of [AD, HD*NH] we initialize [AD, HD*Nkv] layers

    original kv_projection:
    k = W_k(x) == x*w_k --> [BD, CD, ED]*[AD, HD*NH] --> [BD, CD, HD*NH] --> [BD, CD, NH, HD] --> [BD, NH, CD, HD]
    v = W_v(x) == x*w_v --> [BD, CD, ED]*[AD, HD*NH] --> [BD, CD, HD*NH] --> [BD, CD, NH, HD] --> [BD, NH, CD, HD]
    Grouped query projection:
    k = W_k(x) == x*w_k --> [BD, CD, ED]*[AD, HD*Nkv] --> [BD, CD, HD*Nkv] --> [BD, CD, Nkv, HD] --> [BD, Nkv, CD, HD]
    v = W_v(x) == x*w_v --> [BD, CD, ED]*[AD, HD*Nkv] --> [BD, CD, HD*Nkv] --> [BD, CD, Nkv, HD] --> [BD, Nkv, CD, HD]
    
    to compute attention: we need q*k^T but q dim is [BD, NH, CD, HD] and k dim is [BD, Nkv, CD, HD]
    we need to replicate k to NH/Nkv times stack it adn then compute q*k^T
    Group size = GS = NH/Nkv
    k = k.repeat_interleave(GS, dim=1) --> [BD, Nkv*GS, CD, HD] --> [BD, NH, CD, HD]
    v = v.repeat_interleave(GS, dim=1) --> [BD, Nkv*GS, CD, HD] --> [BD, NH, CD, HD]
    we are still using only Nkv kv projections(weights) but replicating(sharing) them for ease of matrix multiplication
    during backpropagation, with in a single kv group for GS queries all the GS gradients are accumulated

    A = softmax(q*k^T/sqrt(HD)) --> [BD, NH, CD, HD]*[BD, NH, CD, HD]^T  --> [BD, NH, CD, CD]
    Mask = torch.triu(torch.ones(1, 1, CD, CD), diagonal=1)
    Mask.masked_fill_(Mask.bool(), -torch.inf) # filling upper triangle above diaognal with -inf, so that softmax will ignore those values
    A = A*Mask
    y = A*v --> [BD, NH, CD, CD]*[BD, NH, CD, HD] --> [BD, NH, CD, HD] -> [BD, CD, NH, HD] -> [BD, CD, HD*NH]
    out_proj = W_o(y) == y*w_o --> [BD, CD, HD*NH]*[HD*NH, HD*NH] --> [BD, CD, HD*NH]
    y = out_proj(y) --> [BD, CD, HD*NH]*[HD*NH, HD*NH] --> [BD, CD, HD*NH]
    HD*NH should be equal to ED
    '''
    def __init__(self, params: LLMParams):
        super().__init__()
        self.params = params
        if params.att_dim % params.num_heads != 0:
            raise ValueError("att_dim must be divisible by num_heads.")
        if params.num_heads % params.num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups.")
        self.group_size = params.num_heads // params.num_kv_groups
        self.q_proj = nn.Linear(params.att_dim, params.head_dim * params.num_heads, bias=params.qkv_bias)
        self.k_proj = nn.Linear(params.att_dim, params.head_dim * params.num_kv_groups, bias=params.qkv_bias)
        self.v_proj = nn.Linear(params.att_dim, params.head_dim * params.num_kv_groups, bias=params.qkv_bias)
        # self.dropout = nn.Dropout(params.dropout)
        self.out_proj = nn.Linear(params.head_dim * params.num_heads, params.head_dim * params.num_heads)
        mask, cos, sin = SharedBuffers.get_buffers(params.context_length, params.head_dim, params.rope_base)
        self.register_buffer('mask', mask)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)


    def forward(self, x):
        bd, cd, ed = x.shape
        q = self.q_proj(x) # [BD, CD, HD*NH]
        k = self.k_proj(x) # [BD, CD, HD*Nkv]
        v = self.v_proj(x) # [BD, CD, HD*Nkv]
        # [BD, CD, NH*HD] --> [BD, CD, NH, HD] --> [BD, NH, CD, HD]
        q = q.view(bd, cd, self.params.num_heads, self.params.head_dim).transpose(1, 2) # [BD, NH, CD, HD]
        k = k.view(bd, cd, self.params.num_kv_groups, self.params.head_dim).transpose(1, 2) # [BD, Nkv, CD, HD]
        v = v.view(bd, cd, self.params.num_kv_groups, self.params.head_dim).transpose(1, 2) # [BD, Nkv, CD, HD]
        q = compute_rope(q, self.cos, self.sin)
        k = compute_rope(k, self.cos, self.sin)
        k = k.repeat_interleave(self.group_size, dim=1) # [BD, NH, CD, HD]
        v = v.repeat_interleave(self.group_size, dim=1) # [BD, NH, CD, HD]
        a = torch.matmul(q, k.transpose(-2, -1)) # [BD, NH, CD, CD]
        a.masked_fill_(self.mask.bool()[:cd, :cd], -torch.inf)
        a = torch.softmax(a / (self.params.head_dim ** 0.5), dim=-1)
        # a = self.dropout(a)
        y = torch.matmul(a, v)
        y = y.transpose(1, 2).contiguous().view(bd, cd, self.head_dim*self.params.num_heads)
        y = self.out_proj(y)
        return y


class CausalMultiHeadFlashAttention(nn.Module):
    '''
    Multi Head Self Attention w/o head sequence
    x --> [BD, CD, ED]
    num_heads = NH
    q = W_q(x) == x*w_q --> [BD, CD, ED]*[AD, HD*NH] --> [BD, CD, HD*NH] --> [BD, CD, NH, HD] --> [BD, NH, CD, HD]
    (ED = AD, HD = ED/NH)
    k = W_k(x) == x*w_k --> [BD, CD, ED]*[AD, HD*NH] --> [BD, CD, HD*NH] --> [BD, CD, NH, HD] --> [BD, NH, CD, HD]
    v = W_v(x) == x*w_v --> [BD, CD, ED]*[AD, HD*NH] --> [BD, CD, HD*NH] --> [BD, CD, NH, HD] --> [BD, NH, CD, HD]
    out_proj = W_o(y) == y*w_o --> [BD, CD, HD*NH]*[HD*NH, HD*NH] --> [BD, CD, HD*NH]

    replicating num of heads attn 
    A = softmax(q*k^T/sqrt(HD)) --> [BD, NH, CD, HD]*[BD, NH, CD, HD]^T  --> [BD, NH, CD, CD]
    Mask = torch.triu(torch.ones(1, 1, CD, CD), diagonal=1)
    Mask.masked_fill_(Mask.bool(), -torch.inf) # filling upper triangle above diaognal with -inf, so that softmax will ignore those values
    A = A*Mask
    y = A*v --> [BD, NH, CD, CD]*[BD, NH, CD, HD] --> [BD, NH, CD, HD] -> [BD, CD, NH, HD] -> [BD, CD, HD*NH]
    y = out_proj(y) --> [BD, CD, HD*NH]*[HD*NH, HD*NH] --> [BD, CD, HD*NH]
    HD*NH should be equal to ED
    '''
    def __init__(self, params):
        super().__init__()
        self.params = params
        if params.att_dim % params.num_heads != 0:
            raise ValueError("att_dim must be divisible by num_heads.")
        self.q_proj = nn.Linear(params.att_dim, params.head_dim * params.num_heads, bias=params.qkv_bias)
        self.k_proj = nn.Linear(params.att_dim, params.head_dim * params.num_heads, bias=params.qkv_bias)
        self.v_proj = nn.Linear(params.att_dim, params.head_dim * params.num_heads, bias=params.qkv_bias)
        self.register_buffer('mask', torch.triu(torch.ones(params.context_length, params.context_length), diagonal=1))
        self.dropout = nn.Dropout(params.dropout)
        self.out_proj = nn.Linear(params.head_dim * params.num_heads, params.head_dim * params.num_heads)
        self.num_heads = params.num_heads
        cos, sin = precompute_rope(params.context_length, params.head_dim, params.rope_base)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def forward(self, x):
        bd, cd, ed = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        head_dim = q.shape[-1] // self.num_heads
        # [BD, CD, NH*HD] --> [BD, CD, NH, HD] --> [BD, NH, CD, HD]
        q = q.view(bd, cd, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(bd, cd, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(bd, cd, self.num_heads, head_dim).transpose(1, 2)
        use_dropout = 0 if self.training else self.dropout
        # if is_causal: False, have to provide mask
        q = compute_rope(q, self.cos, self.sin)
        k = compute_rope(k, self.cos, self.sin)
        y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout=use_dropout, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bd, cd, head_dim*self.num_heads)
        y = self.out_proj(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.fc1 = nn.Linear(params.att_dim, params.ffn_dim, dtype=params.dtype, bias=False)
        self.fc2 = nn.Linear(params.att_dim, params.ffn_dim, dtype=params.dtype, bias=False)
        self.fc3 = nn.Linear(params.ffn_dim, params.att_dim, dtype=params.dtype, bias=False)

    def forward(self, x):
        return self.fc3(nn.functional.silu(self.fc1(x)) * self.fc2(x))

class TransformerBlock(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.att = CausalMultiHeadGroupAttention(params)
        self.ff = FeedForward(params)
        # self.ff = nn.Sequential(
        #     nn.Linear(att_dim, ffn_dim, dtype=dtype, bias=False),
        #     nn.GELU(),
        #     nn.Linear(ffn_dim, att_dim, dtype=dtype, bias=False)
        # )
        self.norm1 = nn.RMSNorm(params.att_dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(params.att_dim, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # y = self.attention(self.norm1(x))
        # x = x + self.dropout(y)
        x = x + self.att(self.norm1(x))
        # y = self.ff(self.norm2(x))
        # x = x + self.dropout(y)
        x = x + self.ff(self.norm2(x))
        return x

class Llama32Model(nn.Module):
    def __init__(self, params: LLMParams):
        super().__init__()
        self.params = params
        self.tok_emb = nn.Embedding(params.vocab_size, params.embed_dim)
        # self.pos_emb = nn.Embedding(params.context_length, params.embed_dim)
        self.dropout = nn.Dropout(params.dropout)
        self.trf_blocks = nn.Sequential(*[
            TransformerBlock(params)
            for _ in range(params.num_layers)
        ])
        self.final_norm = nn.RMSNorm(params.embed_dim, eps=1e-6)
        self.out_head = nn.Linear(params.embed_dim, params.vocab_size, bias=False)
        # Tie weights
        self.out_head.weight = self.tok_emb.weight

    def forward(self, tokens):
        bd, cd = tokens.shape
        x = self.tok_emb(tokens) #+ self.pos_emb(torch.arange(cd, device=tokens.device))
        x = self.trf_blocks(x, x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
