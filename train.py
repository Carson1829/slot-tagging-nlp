import torch
import torch.nn as nn
from torch.optim import Adam

def train_model(model, dataloader, tag2idx, epochs=10, lr=1e-3, has_attention=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tag2idx["<PAD>"])

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y, mask in dataloader:
            x, y = x.to(device), y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()

            logits = model(x)  # (B, L, C(tag_size))
            B, L, C = logits.shape

            logits = logits.reshape(B * L, C)
            y = y.reshape(B * L)
            mask = mask.reshape(B * L)

            if has_attention:
                active_logits = logits[mask == 1]
                active_labels = y[mask == 1]
                loss = criterion(active_logits, active_labels)
            else:
                loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model