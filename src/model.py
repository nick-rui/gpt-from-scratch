import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer import TransformerBlock

class LanguageModel(nn.Module):
    '''
    A transformer-based language model.
    The forward pass gives the standard model outputs (logits) and loss (if targets provided)
    The generate function uses the forward pass and logits of the last index to generate token sequences.
    '''

    def __init__(self, vocab_size, d_emb, d_head, d_mlp, max_context_length, num_heads, num_blocks):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_context_length = max_context_length
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb)
        self.position_embedding = nn.Embedding(num_embeddings=max_context_length, embedding_dim=d_emb)
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_emb, d_head, d_mlp, max_context_length, num_heads) for _ in range(num_blocks)]
        )
        self.unembedding = nn.Linear(in_features=d_emb, out_features=vocab_size)
    
    def forward(self, tokens, labels=None):
        '''
        Returns logits. The i-th entry represents logits for the i+1th token. 
        
        Input (tokens):  (B, T)
        Output (logits): (B, T, V)

        B = (collapsed) batch dimension
        T = number of tokens
        V = vocab size
        '''
        T = tokens.shape[-1]
        assert T <= self.max_context_length
        x = self.token_embedding(tokens) + self.position_embedding(torch.arange(T, device=self.device))
        x = self.blocks(x)
        logits = self.unembedding(x)
        
        if labels is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            labels = labels.view(B * T)
            loss = F.cross_entropy(input=logits, target=labels)
        
        return logits, loss
    
    def generate(self, context, max_tokens_generated):
        running_context = context    # (B, T)
        for _ in range(max_tokens_generated):
            relevant_context = running_context[:, -min(len(running_context), self.max_context_length):]
            logits, _ = self(tokens=relevant_context)
            next_token_logits = logits[:, -1, :]                                # (B, V)
            next_token_probs = F.softmax(next_token_logits, dim=-1)             # (B, V)
            next_token = torch.multinomial(next_token_probs, num_samples=1)     # (B, 1)
            running_context = torch.cat((running_context, next_token), dim=-1)  # (B, T) + (B, 1) = (B, T + 1)
        
        return running_context
    
    
        



