import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionHead(nn.Module):
    '''
    A single attention head without projection back to the residual stream dimension.
    (Projection is ommited here to be compatible for multi-head attention implementations)

    We use scaled dot-product attention. 

    Input:  (B, T, d_emb)
    Output: (B, T, d_head)

    B = batch dimension (collapsed into one)
    T = number of tokens in context (T <= max_context_length)
    d_emb = embedding dimension of model
    d_head = size of attention head
    '''
    def __init__(self, d_emb, d_head, max_context_length):
        super().__init__()
        self.d_emb = d_emb
        self.d_head = d_head
        self.key = nn.Linear(in_features=d_emb, out_features=d_head, bias=False)
        self.query = nn.Linear(in_features=d_emb, out_features=d_head, bias=False)
        self.value = nn.Linear(in_features=d_emb, out_features=d_head, bias=False)
        self.register_buffer(name='tril', tensor=torch.tril(torch.ones(max_context_length, max_context_length)))
    
    def forward(self, x):
        T = x.shape[-2]  # First few dimensions all collapse to be "batch dimension" B

        k = self.key(x)   # (B, T, d_head)
        q = self.query(x) # (B, T, d_head)
        v = self.value(x) # (B, T, d_head)

        a = q @ k.transpose(-1, -2) * (self.d_head ** -0.5)   # (B. T, T)
        a = a.masked_fill(self.tril[:T, :T] == 0, float('-inf'))   # (B, T, T)
        a = F.softmax(a, dim=-1)        # (B, T, T)

        output = a @ v   # (B, T, d_head)
        return output

class MultiHeadAttention(nn.Module):
    '''
    A layer of multiple heads of attention ran independently followed by a projection back to the residual stream dimension.

    Input:  (B, T, d_emb)
    Output: (B, T, d_emb)

    B = (collapsed) batch dimension
    T = number of tokens (<= max_context_length)
    d_emb = embedding dimension
    '''
    def __init__(self, d_emb, d_head, max_context_length, num_heads):
        super().__init__()
        self.d_emb = d_emb
        self.num_heads = num_heads
        self.d_heads = num_heads * d_head
        self.heads = nn.ModuleList([AttentionHead(d_emb, d_head, max_context_length) for _ in range(num_heads)])
        self.proj = nn.Linear(in_features=self.d_heads, out_features=d_emb)
    
    def forward(self, x):
        T = x.shape[-2]
        output = torch.cat([head(x) for head in self.heads], dim=-1)  # num_heads x (B, T, d_head) -> (B, T, d_heads)
        output = self.proj(output)  # (B, T, d_emb)
        return output
    
class MultilayerPerceptron(nn.Module):
    '''
    A multilayer perceptron (feed forward neural network) consisting of 
    1) up projection from d_emb -> d_mlp
    2) nonlinear function (ReLU)
    3) down projection from d_mlp -> d_emb (back to residual stream)

    Input:  (B, T, d_emb)
    Output: (B, T, d_emb)

    B = (collapsed) batch dimension
    T = number of tokens
    d_emb = embedding dimension of model
    '''
    def __init__(self, d_emb, d_mlp):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=d_emb, out_features=d_mlp),
            nn.ReLU(),
            nn.Linear(in_features=d_mlp, out_features=d_emb)
        )
    
    def forward(self, x):
        output = self.mlp(x)
        return output

class TransformerBlock(nn.Module):
    '''
    A single transformer block which the residual stream runs through. 
    We utilize layer normalization here as well.

    Input:  (B, T, d_emb)  read from residual stream
    Output: (B, T, d_emb)  write to residual stream
    '''
    def __init__(self, d_emb, d_head, d_mlp, max_context_length, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_emb, d_head, max_context_length, num_heads)
        self.mlp = MultilayerPerceptron(d_emb, d_mlp)
        self.ln1 = nn.LayerNorm(d_emb)
        self.ln2 = nn.LayerNorm(d_emb)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.attn(x)
        x = self.ln2(x)
        x = x + self.mlp(x)
        output = x
        return output

