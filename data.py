import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
import torch


def tokenize(sentences):
    # Split sentences by whitespace (simple word-level tokenizer)
    return [s.split() for s in sentences]


def load_data(df, has_tags=False):
    """
    Reads sentences and (optional) tag sequences from dataframe.
    """
    if has_tags:
        sentences = df["utterances"].tolist()
        tags = [t.split() for t in df["IOB Slot tags"].tolist()]
    else:
        sentences = df["utterances"].tolist()
        tags = None
    return sentences, tags


def build_vocab(tokenized_sents, min_freq=1):
    """
    Builds vocabulary from tokenized sentences.
    """
    counter = Counter()
    for sent in tokenized_sents:
        counter.update(sent)

    # Special tokens FIRST for consistent indexing
    vocab = ["<PAD>", "<UNK>"] + [w for w, c in counter.items() if c >= min_freq]
    
    # Mapping word → index
    word2idx = {w: i for i, w in enumerate(vocab)}
    return word2idx, vocab


def build_tag_vocab(tag_lists):
    """
    Build tag vocabulary (IOB tags + <PAD>).
    """
    unique_tags = set()
    for tags in tag_lists:
        unique_tags.update(tags)

    tag_list = ["<PAD>"] + sorted(list(unique_tags))
    tag2idx = {t: i for i, t in enumerate(tag_list)}
    return tag2idx, tag_list


def pad_sequences(seqs, pad_value=0):
    """
    Pads sequences to the longest sequence in the batch.
    Returns:
        padded sequences (tensor)
        mask tensor (1 = real token, 0 = padding)
    """
    max_len = max(len(s) for s in seqs)
    padded = []
    masks = []

    for s in seqs:
        padded.append(s + [pad_value] * (max_len - len(s)))
        masks.append([1] * len(s) + [0] * (max_len - len(s)))

    return torch.tensor(padded), torch.tensor(masks)


class DS(Dataset):
    """
    PyTorch Dataset for sequence tagging.
    Converts tokens → indices
    """
    def __init__(self, sentences, tags, word2idx=None, tag2idx=None, build_vocab=False):
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Convert tokens to IDs
        words = self.sentences[idx]
        x = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]

        # Convert tags to IDs
        if self.tags is None:
            y = None
        else:
            y = [self.tag2idx[t] for t in self.tags[idx]]

        return x, y


def collate_fn(batch):
    """
    Pads token and tag sequences for a batch.
    Used by DataLoader.
    """
    xs, ys = zip(*batch)

    padded_x, masks = pad_sequences(xs, pad_value=0)

    # If no labels (test set), create dummy labels
    if ys[0] is None:
        padded_y = torch.zeros_like(padded_x)
    else:
        padded_y, _ = pad_sequences(ys, pad_value=0)

    return padded_x, padded_y, masks