

import pandas as pd
import numpy as np


def tokenize(sentence: str):
    return sentence.strip().split()


def build_vocab(sentences, min_freq=1):
    freq = {}
    for s in sentences:
        for w in s:
            freq[w] = freq.get(w, 0) + 1
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    idx = 4
    for w, c in freq.items():
        if c >= min_freq and w not in vocab:
            vocab[w] = idx
            idx += 1
    ivocab = {i: w for w, i in vocab.items()}
    return vocab, ivocab


def numericalize(sentences, vocab, max_len=50):
    sos_idx = vocab['<sos>']
    eos_idx = vocab['<eos>']
    pad_idx = vocab['<pad>']
    # <unk> already in vocab for unknown words

    num_sentences = []
    for s in sentences:
        s = [w if w in vocab else '<unk>' for w in s]
        s = [sos_idx] + [vocab[w] for w in s] + [eos_idx]
        if len(s) < max_len:
            s = s + [pad_idx] * (max_len - len(s))
        else:
            s = s[:max_len]
        num_sentences.append(s)
    return num_sentences


def preprocess_data(csv_path, src_col='eng', tgt_col='darija_ar', max_len=50, train_split=0.8):
    data = pd.read_csv(csv_path)

    def clean_text(text):
        return text.strip()

    src_sentences = [tokenize(clean_text(t)) for t in data[src_col].astype(str).tolist()]
    tgt_sentences = [tokenize(clean_text(t)) for t in data[tgt_col].astype(str).tolist()]

    src_vocab, src_ivocab = build_vocab(src_sentences, min_freq=1)
    tgt_vocab, tgt_ivocab = build_vocab(tgt_sentences, min_freq=1)

    src_num = numericalize(src_sentences, src_vocab, max_len)
    tgt_num = numericalize(tgt_sentences, tgt_vocab, max_len)

    total = len(src_num)
    idxs = np.arange(total)
    np.random.shuffle(idxs)
    train_size = int(train_split * total)
    train_idx = idxs[:train_size]
    val_idx = idxs[train_size:]

    src_train = [src_num[i] for i in train_idx]
    tgt_train = [tgt_num[i] for i in train_idx]
    src_val = [src_num[i] for i in val_idx]
    tgt_val = [tgt_num[i] for i in val_idx]

    return (src_train, tgt_train), (src_val, tgt_val), src_vocab, tgt_vocab
