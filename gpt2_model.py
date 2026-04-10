import torch
import torch.nn as nn 

from types import SimpleNamespace
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "out_dim": 768,
    "n_heads": 12, ## Number of heads in a transformer
    "n_layers": 12, ## Number of transformer layers
    "drop_rate": 0.1,
    "qkv_bias": False
}


config = SimpleNamespace(**GPT_CONFIG_124M)
class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True, unbiased=False)
        norm_x = (x-mean)/ torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift


class MultiHeadAttention(nn.Module):
    """
    Analyzes relationship between input elements 
    """
    def __init__(self,cfg):
        super().__init__()
        
        self.out_dim = cfg.out_dim
        self.num_heads = cfg.n_heads
        self.per_head_out_dim = self.out_dim//self.num_heads

    
        self.query_matrix = nn.Linear(cfg.emb_dim,cfg.out_dim,bias=cfg.qkv_bias)
        self.key_matrix = nn.Linear(cfg.emb_dim,cfg.out_dim,bias=cfg.qkv_bias)
        self.value_matrix = nn.Linear(cfg.emb_dim,cfg.out_dim,bias=cfg.qkv_bias)

        self.drop_out = nn.Dropout(cfg.drop_rate)
        self.register_buffer("mask",
            torch.triu(torch.ones(cfg.context_length,cfg.context_length),diagonal=1)
        )

        self.out_proj = nn.Linear(self.out_dim, self.out_dim)


    def forward(self,x):
        batch_size, num_input_tokens, emb_dim = x.shape
        
        ### Shape = batch_size * num_input_tokens * out_dim
        queries = self.query_matrix(x)
        keys = self.key_matrix(x)
        values = self.value_matrix(x)


        ### Attention needs to be applied head wise, so split into heads
        queries = queries.view(batch_size, num_input_tokens, self.num_heads, self.per_head_out_dim)
        keys = keys.view(batch_size, num_input_tokens, self.num_heads, self.per_head_out_dim)
        values = values.view(batch_size, num_input_tokens, self.num_heads, self.per_head_out_dim)

        ### Apply transpose to push num_input_tokens , per_head_out to end
        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        ### Compute Attention
        #1. Attention Scores
        attn_scores = queries @ keys.transpose(2,3)

        #2. Masking
        mask_bool = self.mask.bool()[:num_input_tokens,:num_input_tokens]
        attn_scores.masked_fill_(mask_bool,-torch.inf)

        #3. Attention Weights
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)

        #4. Dropout
        attn_weights=self.drop_out(attn_weights)

        #5. Context Vector
        context_vec = (attn_weights @ values).transpose(1,2)

        #6. Reshape Context Vector back
        context_vec=context_vec.contiguous().view(batch_size,num_input_tokens,self.out_dim)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec



class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5 * x * (

            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0/torch.pi)) * 
                (x+0.044715 * torch.pow(x,3))
            )
        )

class FeedForwardNN(nn.Module):
    """
    Handles each token independently unlike attention
    """
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4*cfg.emb_dim), ## Expansion
            GELU(),
            nn.Linear(4*cfg.emb_dim,cfg.emb_dim),  ## Contraction
        )
    
    def forward(self,x):
        return self.layers(x)


class Transformer(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        ## Layer Normalization
        # These are called pre-LayerNorm layers as they are applied before attention and feedforward pass
        # Older version used to have pro_LayerNorm but that is found to have wrose training dynamics
        self.layer_norm1 = LayerNorm(cfg.emb_dim)
        self.layer_norm2 = LayerNorm(cfg.emb_dim)

        ## Mulit Head Attention
        self.multi_head_attention = MultiHeadAttention(cfg)

        ## FeedForward Neural Network
        self.feed_forward_nn = FeedForwardNN(cfg) 

        ## Dropout 
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)
    

    def forward(self,x):

        # Shortcut connection for attention block
        shortcut = x
        x = self.layer_norm1(x)
        x = self.multi_head_attention(x) # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x= x+shortcut # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward_nn(x)
        x = self.drop_shortcut(x)
        x = x+shortcut # Add the original input back

        return x

class GPT2(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(
            cfg.vocab_size,cfg.emb_dim
        )

        self.pos_emb = nn.Embedding(
            cfg.context_length,cfg.emb_dim
        )

        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(
            *[Transformer(cfg) for _ in range(cfg.n_layers)]
        )

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(
            cfg.emb_dim,cfg.vocab_size,bias = False
        )


    def forward(self,x):
        batch_size,seq_len = x.shape
        
        ### Embeddings
        embeddings = self.tok_emb(x)
        pos_embeddings = self.pos_emb(torch.arange(seq_len,device = x.device))

        final_input_embeddings = embeddings + pos_embeddings


        ### Dropout
        # - Randomly turnoff input embeddings to 0
        # prevents overfitting
        x = self.drop_emb(final_input_embeddings)


        ### Transformer
        x = self.trf_blocks(x)

        ## Layer Normalization
        x = self.final_norm(x)

        ## getting logits out of the model
        ## siz eof this is num_tokens * vocab_size
        logits = self.out_head(x)

        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context

    ###Input batch:
 ###tensor([[6109, 3626, 6100,  345],
        ##[6109, 1110, 6622,  257]])
    
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond) ### batch, n_tokens, vocab_size
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


import tiktoken
tokenizer = tiktoken.get_encoding('gpt2')

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

    
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx