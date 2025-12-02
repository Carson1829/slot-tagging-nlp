import pandas as pd
from collections import Counter


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

def encode_sentences(tokenized_sents, word2idx):
    encoded = []
    for sent in tokenized_sents:
        encoded.append([word2idx.get(w, word2idx["<UNK>"]) for w in sent])
    return encoded

def encode_tags(tag_lists, tag2idx):
    return [[tag2idx[t] for t in tags] for tags in tag_lists]

def pad_sequences(seqs, pad_value=0):
    max_len = max(len(s) for s in seqs)
    padded = []
    masks = []
    for s in seqs:
        padded.append(s + [pad_value] * (max_len - len(s)))
        masks.append([1] * len(s) + [0] * (max_len - len(s)))
    return torch.tensor(padded), torch.tensor(masks)