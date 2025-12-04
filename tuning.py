import pandas as pd
import torch
from data import load_data, tokenize, build_vocab, build_tag_vocab, DS, collate_fn
from torch.utils.data import DataLoader
from model import RNNs
from train import train_val_loop
from sklearn.model_selection import train_test_split
import random


def search(tag_list, train_loader, val_loader, vocab_size, tag2idx, model_types, hidden_sizes, dropouts, 
           num_layers, attention_heads_list, lrs, emb_dim=100, num_trials=30):
    # Track the best validation F1 score found across all trials
    best_f1 = 0.0
    best_config = None          # store the hyperparameter set that achieved best F1
    best_model_state = None     # store the corresponding model weights

    # Run randomized hyperparameter search
    for i in range(num_trials):

        # --- Sample a random hyperparameter configuration ---
        config = {
            "model_type": random.choice(model_types),             # choose RNN, GRU, LSTM
            "hidden_dim": random.choice(hidden_sizes),            # choose hidden size
            "dropout": random.choice(dropouts),                   # choose dropout rate
            "num_layers": random.choice(num_layers),              # choose number of RNN layers
            "attention_heads": random.choice(attention_heads_list),  # choose number of attention heads
            "LRs": random.choice(lrs)                             # choose a learning rate (not used below!)
        }

        print("Trial config:", config)

        # --- Build model for this trial ---
        model = RNNs(
            vocab_size=vocab_size,
            tag_size=len(tag2idx),
            model_type=config["model_type"],
            n_layers=config["num_layers"],
            emb_dim=emb_dim,
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            attention_heads=config["attention_heads"]
        )

        # --- Train + validate ---
        # train_val_loop trains with early stopping and returns:
        #   - best model state for this trial
        #   - best F1 obtained on validation set
        best_state, f1 = train_val_loop(
            model,
            tag_list,
            train_loader,
            val_loader,
            tag2idx,
            lr=1e-3,          # NOTE: fixed LR here, even though a sampled LR exists in config["LRs"]
            epochs=20
        )

        # --- Update global best result if this trial outperformed previous trials ---
        if f1 > best_f1:
            best_f1 = f1
            best_config = config
            best_model_state = best_state

    # After all trials, print summary of top-performing hyperparameters
    print("Best Config:", best_config)
    print("Best Val F1:", best_f1)

    # Return weights & configuration of best model
    return best_model_state, best_config, best_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CSV
    initial_df = pd.read_csv("data/train.csv")

    train_data, val_data = train_test_split(initial_df,random_state = 32, test_size = 0.25, shuffle = True)

    # Load sentences and tags
    train_sents, train_tags = load_data(train_data, has_tags=True)
    val_sents, val_tags = load_data(val_data, has_tags=True)

    # Tokenize
    tokenized_train = tokenize(train_sents)
    tokenized_val = tokenize(val_sents)

    # Build vocabularies
    word2idx, vocab = build_vocab(tokenized_train)
    tag2idx, tag_list = build_tag_vocab(train_tags)

    # Create datasets
    train_dataset = DS(tokenized_train, train_tags, word2idx=word2idx, tag2idx=tag2idx)
    val_dataset = DS(tokenized_val, val_tags, word2idx=word2idx, tag2idx=tag2idx)

    # create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # create model
    vocab_size = len(word2idx)
    tag_size = len(tag2idx)

    # hyperparameter search field
    model_types = ["gru", "lstm"]
    hidden_sizes = [128, 256]
    dropouts = [0.1, 0.3]
    lrates = [0.001, 0.003]
    num_layers = [2, 3, 4]
    attention_heads_list = [0, 2, 4]  # 0 means no attention

    best_state, best_config, best_f1 = search(tag_list, train_loader, val_loader, 
        vocab_size, tag2idx, model_types, hidden_sizes, dropouts,
        num_layers, attention_heads_list, lrates
    )
    print(f"F1: {best_f1}, Config: {best_config}")


if __name__ == "__main__":
    main()