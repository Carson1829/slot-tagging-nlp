import torch
import torch.nn as nn
from torch.optim import Adam
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
import copy

def train_model(model, dataloader, tag2idx, epochs=10, lr=0.001):
    '''
    Training on the full training set for final submission
    Outputs fully trained model
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimizer and Loss function
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tag2idx["<PAD>"])

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            x, y, mask = batch
            x, y = x.to(device), y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()

            logits = model(x, mask)  # (B, L, C(tag_size))
            B, L, C = logits.shape

            preds = logits.reshape(B * L, C)
            y = y.reshape(B * L)
            loss = loss_fn(preds, y)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model


def train_val_loop(model, tag_list, train_loader, val_loader, tag2idx, epochs=10, lr=0.001, attention_heads=0):
    '''
    Training and validation for hyperparameter tuning
    Outputs the best validation F1 score and best model state
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tag2idx["<PAD>"])

    best_val_f1 = 0.0
    best_model = None
    counter = 0
    patience = 2

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0
        for batch in train_loader:
            x, y, mask = batch
            x, y = x.to(device), y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            logits = model(x, mask)  # (B, L, C(tag_size))
            B, L, C = logits.shape
            preds = logits.reshape(B*L, C)
            y = y.reshape(B*L)
            
            loss = loss_fn(preds, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # validation loop
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, y, mask = batch
                x, y = x.to(device), y.to(device)
                mask = mask.to(device)
                logits = model(x, mask)
                B, L, C = logits.shape
                preds = logits.argmax(dim=-1)
                # iterate each sentence
                for i in range(len(preds)):
                    seq_len = int(mask[i].sum().item())

                    # predicted tags
                    pred_seq = preds[i][:seq_len].tolist()
                    pred_tags = [tag_list[int(idx)] for idx in pred_seq]

                    # true tags
                    true_seq = y[i][:seq_len].tolist()
                    true_tags = [tag_list[int(idx)] for idx in true_seq]

                    all_preds.append(pred_tags)
                    all_labels.append(true_tags)

        val_f1 = f1_score(all_labels, all_preds, mode="strict", scheme=IOB2)
        print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, Val F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    print(f"Best Validation F1: {best_val_f1:.4f}")
    return best_model, best_val_f1