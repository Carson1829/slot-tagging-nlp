import torch
import pandas as pd
from data import load_data, tokenize, build_vocab


def load_glove_full(path, dim):
    glove = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
            glove[word] = vector
    return glove

def build_trimmed_embedding_matrix(word2idx, glove, emb_dim):
    vocab_size = len(word2idx)
    emb_matrix = torch.zeros((vocab_size, emb_dim))

    for word, idx in word2idx.items():
        if word in glove:
            emb_matrix[idx] = glove[word]
        else:
            # random init for words not in glove
            emb_matrix[idx] = torch.randn(emb_dim) * 0.1

    return emb_matrix

def main():
    glove_path = "data/glove.6B.100d.txt"
    emb_dim = 100

    train_data = pd.read_csv("train.csv")
    train_sents, train_tags = load_data(train_data, has_tags=True)
    tokenized_train = tokenize(train_sents)
    word2idx, vocab = build_vocab(tokenized_train)

    # load full glove
    glove = load_glove_full(glove_path, emb_dim)

    # build trimmed matrix
    embedding_matrix = build_trimmed_embedding_matrix(word2idx, glove, emb_dim)

    # save trimmed file
    torch.save(embedding_matrix, "embedding.pt")

    print("Saved trimmed GloVe embeddings to embedding.pt!")


if __name__ == "__main__":
    main()