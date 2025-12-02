import torch



def train_model(model, dataloader, save_path='best_model.pt', pad_token_id=None, num_epochs=None):
    
    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)