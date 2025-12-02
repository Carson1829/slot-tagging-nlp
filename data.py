import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
import torch


def tokenize(sentences):
    return [s.split() for s in sentences]

def load_data(df, has_tags = False):
    if has_tags:
        sentences = df["utterances"].tolist()
        tags = [t.split() for t in df["IOB Slot tags"].tolist()]
    else:
        sentences = df["utterances"].tolist()
        tags = None
    return sentences, tags

def build_vocab(tokenized_sents, min_freq=1):
    counter = Counter()
    for sent in tokenized_sents:
        counter.update(sent)

    vocab = ["<PAD>", "<UNK>"] + [w for w, c in counter.items() if c >= min_freq]
    word2idx = {w: i for i, w in enumerate(vocab)}
    return word2idx, vocab

def build_tag_vocab(tag_lists):
    unique_tags = set()
    for tags in tag_lists:
        unique_tags.update(tags)

    tag_list = ["<PAD>"] + sorted(list(unique_tags))
    tag2idx = {t: i for i, t in enumerate(tag_list)}
    return tag2idx, tag_list

def pad_sequences(seqs, pad_value=0):
    max_len = max(len(s) for s in seqs)
    padded = []
    masks = []
    for s in seqs:
        padded.append(s + [pad_value] * (max_len - len(s)))
        masks.append([1] * len(s) + [0] * (max_len - len(s)))
    return torch.tensor(padded), torch.tensor(masks)


class DS(Dataset):
    def __init__(self, sentences, tags, word2idx=None, tag2idx=None, build_vocab=False):
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        x = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]

        if self.tags is None:
            y = None
        else:
            y = [self.tag2idx[t] for t in self.tags[idx]]

        return x, y

# Collate function for DataLoader
def collate_fn(batch):
    xs, ys = zip(*batch)
    padded_x, masks = pad_sequences(xs, pad_value=0)
    if ys[0] is None:
        padded_y = torch.zeros_like(padded_x)
    else:
        padded_y, _ = pad_sequences(ys, pad_value=0)
    return padded_x, padded_y, masks