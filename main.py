import pandas as pd
import torch
from data import load_data, tokenize, build_vocab, build_tag_vocab, DS, collate_fn
from torch.utils.data import DataLoader
from model import RNNs
from train import train_model
from sklearn.model_selection import train_test_split
from evaluation import get_preds
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path")
    parser.add_argument("test_path")
    parser.add_argument("output_csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CSVs
    train_data = pd.read_csv(args.train_path)
    test_data = pd.read_csv(args.test_path)

    # Load sentences and tags
    train_sents, train_tags = load_data(train_data, has_tags=True)
    test_sents, _ = load_data(test_data, has_tags=False)

    # Tokenize
    tokenized_train = tokenize(train_sents)
    tokenized_test = tokenize(test_sents)

    # Build vocabularies
    word2idx, vocab = build_vocab(tokenized_train)
    tag2idx, tag_list = build_tag_vocab(train_tags)

    # Create datasets
    train_dataset = DS(tokenized_train, train_tags, word2idx=word2idx, tag2idx=tag2idx)
    test_dataset = DS(tokenized_test, tags=None, word2idx=word2idx)

    # create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # create model
    vocab_size = len(word2idx)
    tag_size = len(tag2idx)
    emb_matrix = torch.load("embedding.pt")
    model = RNNs(vocab_size=vocab_size, tag_size=tag_size, model_type="lstm", attention_heads=2, n_layers=2, hidden_dim=256, dropout=0.1, pretrained_emb=emb_matrix)

    # train
    model = train_model(model, train_loader, tag2idx, epochs=10, lr=0.001)
    
    outputs = get_preds(model, test_loader, tag_list, device)
    output_df = pd.DataFrame(outputs, columns=["ID","IOB Slot tags"])
    output_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to csv")

if __name__ == "__main__":
    main()
