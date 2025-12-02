import pandas as pd
import torch
from data import load_data, tokenize, build_vocab, build_tag_vocab, DS, collate_fn
from torch.utils.data import DataLoader
from model import RNNs
from train import train_val_loop
from sklearn.model_selection import train_test_split


# -------------------------
# Hyperparameter search example
# -------------------------
def hyperparam_search(tag_list, model_types, hidden_sizes, dropouts, num_layers, attention_heads_list, 
                      train_loader, val_loader, vocab_size, tag2idx, emb_dim=100):

    best_overall_f1 = 0.0
    best_config = None
    best_model_state = None

    for model_type in model_types:
        for hidden_dim in hidden_sizes:
            for dropout in dropouts:
                for n_layer in num_layers:
                    for attn_heads in attention_heads_list:
                        print(f"Training {model_type} | H={hidden_dim} | Dropout={dropout} | Layers={n_layer} | Heads={attn_heads}")
                        model = RNNs(vocab_size, len(tag2idx), model_type, n_layers=n_layer, emb_dim=emb_dim, 
                                         hidden_dim=hidden_dim, dropout=dropout, attention_heads=attn_heads)
                        best_state, best_f1 = train_val_loop(model, tag_list, train_loader, val_loader, tag2idx, lr=1e-3, epochs=20)
                        if best_f1 > best_overall_f1:
                            best_overall_f1 = best_f1
                            best_config = {
                                "model_type": model_type,
                                "hidden_dim": hidden_dim,
                                "dropout": dropout,
                                "num_layers": n_layer,
                                "attention_heads": attn_heads
                            }
                            best_model_state = best_state

    print("Best Config:", best_config)
    print("Best Val F1:", best_overall_f1)
    return best_model_state, best_config, best_overall_f1


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

    model_types = ["rnn", "gru", "lstm"]
    hidden_sizes = [128, 256]
    dropouts = [0.1, 0.3]
    lrates = [1e-3]
    num_layers = [2, 3, 4]
    attention_heads_list = [0, 2, 4]  # 0 means no attention

    best_state, best_config, best_f1 = hyperparam_search(
        tag_list, model_types, hidden_sizes, dropouts, num_layers, attention_heads_list,
        train_loader, val_loader,
        vocab_size, tag2idx,
        emb_dim=100
    )

    # Recreate the model with best hyperparameters
    best_model = RNNs(
        model_type=best_config["model_type"],
        vocab_size=vocab_size,
        tag_size=len(tag2idx),
        n_layers=best_config["num_layers"],
        emb_dim=100,
        hidden_dim=best_config["hidden_dim"],
        dropout=best_config["dropout"],
        attention_heads=best_config["attention_heads"]
    )

    # Load best weights
    best_model.load_state_dict(best_state)
    best_model.to(device)

    # Save
    torch.save(best_model.state_dict(), "best_model.pt")
    print(f"Saved best model with validation F1={best_f1:.4f}")


if __name__ == "__main__":
    main()