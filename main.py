import pandas as pd
import torch
from data import load_data, tokenize, build_vocab, build_tag_vocab, DS, collate_fn
from torch.utils.data import DataLoader
from model import LSTM_NN
from train import train_model
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CSVs
    initial_df = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")

    train_data, val_data = train_test_split(initial_df,random_state = 32, test_size = 0.25, shuffle = True)

    # Load sentences and tags
    train_sents, train_tags = load_data(train_data, has_tags=True)
    val_sents, val_tags = load_data(val_data, has_tags=True)
    test_sents, _ = load_data(test_data, has_tags=False)

    # Tokenize
    tokenized_train = tokenize(train_sents)
    tokenized_val = tokenize(val_sents)
    tokenized_test = tokenize(test_sents)

    # Build vocabularies
    word2idx, vocab = build_vocab(tokenized_train)
    tag2idx, tag_list = build_tag_vocab(train_tags)

    # Create datasets
    train_dataset = DS(tokenized_train, train_tags, word2idx=word2idx, tag2idx=tag2idx)
    val_dataset = DS(tokenized_val, val_tags, word2idx=word2idx, tag2idx=tag2idx)
    test_dataset = DS(tokenized_test, tags=None, word2idx=word2idx)

    # create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # create model
    vocab_size = len(word2idx)
    tag_size = len(tag2idx)
    model = LSTM_NN(vocab_size=vocab_size, tag_size=tag_size)
    model.to(device)

    # train
    train_model(model, train_loader, tag2idx, epochs=10)
    
    model.eval()
    all_preds = []
    all_labels = []


    with torch.no_grad():
        for x, y, mask in val_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1).cpu()

            # iterate each sentence
            for i in range(len(preds)):
                seq_len = int(mask[i].sum().item())

                # predicted tags
                pred_seq = preds[i][:seq_len].tolist()
                pred_tags = [tag_list[idx] for idx in pred_seq]

                # true tags
                gold_seq = y[i][:seq_len].tolist()
                gold_tags = [tag_list[idx] for idx in gold_seq]

                all_preds.append(pred_tags)
                all_labels.append(gold_tags)

    f1 = f1_score(all_labels, all_preds, mode="strict", scheme=IOB2)
    print("F1:", f1)
    # output_df.to_csv("test_pred.csv", index=False)
    # print(f"Predictions saved to output.csv")

if __name__ == "__main__":
    main()
