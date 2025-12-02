import torch
import torch.nn as nn
from torch.optim import Adam

def train_model(model, dataloader, tag2idx, epochs=10, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tag2idx["<PAD>"])

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y, mask in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)  # (B, L, tag_size)
            logits = logits.view(-1, logits.shape[-1])  # (B*L, tag_size)
            y = y.view(-1)                               # (B*L)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model