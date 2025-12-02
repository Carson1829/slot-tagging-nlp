import torch.nn as nn

class LSTM_NN(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim=100, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        # Linear layer to tag space
        self.fc = nn.Linear(hidden_dim * 2, tag_size)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len)
        returns: (batch_size, seq_len, tag_size)
        """
        emb = self.embedding(x)       # (B, L, E)
        lstm_out, _ = self.lstm(emb) # (B, L, 2H)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)   # (B, L, tag_size)
        return logits