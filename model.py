import torch.nn as nn
import torch
import torch.nn.functional as F

# Implementation of a single self-attention head 
class Head(nn.Module):
    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        # linear projections for keys, queries, and values
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size


    def forward(self, x, padding_mask):
        B,S,D = x.shape

        # compute key, query, and value projections
        k = self.key(x)   # (B,S,hs)
        q = self.query(x) # (B,S,hs)
        v = self.value(x) # (B,S,hs)

        # compute attention scores 
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, S, hs) @ (B, hs, S) -> (B, S, S)
        
        # applying padding masks
        NEG = -1e9
        wei = wei.masked_fill(padding_mask[:, None, :] == 0, NEG)

        # normalizing attention weights
        wei = F.softmax(wei, dim=-1) # (B, S, S)
        wei = self.dropout(wei)

        # compute weighted sum of values
        out = wei @ v # (B, S, S) @ (B, S, hs) -> (B, S, hs)
        return out

# Implementation of multi-head self-attention.
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads, dropout):
        super().__init__()
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"
        self.n_heads = n_heads
        head_size = n_embd // n_heads

        # create multiple heads
        self.heads = nn.ModuleList([Head(head_size, n_embd, dropout) for _ in range(n_heads)])
        
        # output projection layer to recombine heads
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        # concatenate outputs from all heads along the last dimension
        out = torch.cat([h(x, padding_mask=padding_mask) for h in self.heads], dim=-1)
        # project concatenated result back to embedding dimension
        out = self.proj(out)
        out = self.dropout(out)
        return out



class RNNs(nn.Module):
    def __init__(self, vocab_size, tag_size, model_type="lstm", attention_heads=0,
                 n_layers=2, emb_dim=100, hidden_dim=256, dropout=0.1,
                 pretrained_emb=None):
        super().__init__()
        self.rnn_type = model_type
        self.attention_heads = attention_heads

        # Embedding layer: maps token IDs → dense vectors
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # Load pretrained embeddings (e.g., GloVe)
        if pretrained_emb is not None:
            # Replace randomly initialized embedding weights
            self.embedding.weight = nn.Parameter(pretrained_emb, requires_grad=True)

        # Create RNN/LSTM/GRU depending on model_type argument
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True   # BiRNN: forward + backward context
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True
            )
        else:  # vanilla RNN
            self.rnn = nn.RNN(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout,
                nonlinearity="tanh",
                batch_first=True,
                bidirectional=True
            )

        # Dropout after RNN for regularization
        self.dropout = nn.Dropout(dropout)

        # Optional multi-head attention layer
        if self.attention_heads > 0:
            # Multi-head self-attention (your implementation)
            self.attn = MultiHeadAttention(hidden_dim * 2, n_heads=self.attention_heads, dropout=0.3)
            self.attn_norm = nn.LayerNorm(hidden_dim * 2)  # Residual + normalization

        # Final classifier — projects hidden states to tag logits
        self.fc = nn.Linear(hidden_dim * 2, tag_size)
    
    def forward(self, x, mask):
        """
        x: (batch_size, seq_len) token IDs
        mask: (batch_size, seq_len) binary mask, 1 = real token, 0 = padding
        """
        # Convert token IDs → embeddings
        emb = self.embedding(x)  # (B, L, E)

        # Run through selected RNN type
        rnn_out, _ = self.rnn(emb)  # (B, L, 2H)
        rnn_out = self.dropout(rnn_out)

        # Optional attention block
        if self.attention_heads > 0:
            padding_mask = (mask == 0)  # Mask: True at padding positions
            attn_out = self.attn(rnn_out, padding_mask)  # Self-attention output
            # Residual connection and layer norm
            rnn_out = self.attn_norm(rnn_out + attn_out)

        # Compute tag logits for each token
        logits = self.fc(rnn_out)  # (B, L, tag_size)
        return logits