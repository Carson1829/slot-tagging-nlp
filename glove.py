import torch
import pandas as pd
from data import load_data, tokenize, build_vocab


def load_glove_full(path, dim):
    """
    Loads the full GloVe embedding file into memory.
    
    Args:
        path (str): Path to the raw GloVe .txt file.
        dim (int): Dimensionality of the word vectors (e.g., 100).
    
    Returns:
        dict: A dictionary mapping each word to its embedding vector (torch.tensor).
    """
    glove = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Split each line into: word + embedding values
            parts = line.rstrip().split(" ")
            word = parts[0]
            vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)

            # Store vector in dictionary
            glove[word] = vector
    return glove


def build_trimmed_embedding_matrix(word2idx, glove, emb_dim):
    """
    Builds an embedding matrix *only for the words in your dataset vocabulary*.
    This cuts down memory use compared to storing the entire GloVe vocabulary.
    
    Args:
        word2idx (dict): Mapping of word -> index from your dataset.
        glove (dict): Full GloVe embedding dictionary (word -> vector).
        emb_dim (int): Dimensionality of embeddings.
    
    Returns:
        torch.Tensor: A (vocab_size, emb_dim) matrix initialized from GloVe
                      when possible, random otherwise.
    """
    vocab_size = len(word2idx)
    
    # Initialize empty embedding matrix
    emb_matrix = torch.zeros((vocab_size, emb_dim))

    for word, idx in word2idx.items():
        if word in glove:
            # Use pretrained GloVe vector
            emb_matrix[idx] = glove[word]
        else:
            # Random initialization for missing words
            emb_matrix[idx] = torch.randn(emb_dim) * 0.1

    return emb_matrix


def main():
    """
    Loads the dataset, builds the vocabulary, loads GloVe, trims it to only
    the dataset's words, and saves the resulting embedding matrix.
    """
    glove_path = "data/glove.6B.100d.txt"
    emb_dim = 100

    # Load training data
    train_data = pd.read_csv("train.csv")
    train_sents, train_tags = load_data(train_data, has_tags=True)

    # Tokenize sentences and build vocabulary from dataset
    tokenized_train = tokenize(train_sents)
    word2idx, vocab = build_vocab(tokenized_train)

    # Load the full GloVe dictionary (large file)
    glove = load_glove_full(glove_path, emb_dim)

    # Build a trimmed embedding matrix only for dataset vocab words
    embedding_matrix = build_trimmed_embedding_matrix(word2idx, glove, emb_dim)

    # Save the final embedding tensor for use in your model
    torch.save(embedding_matrix, "embedding.pt")

    print("Saved trimmed GloVe embeddings to embedding.pt!")


if __name__ == "__main__":
    main()