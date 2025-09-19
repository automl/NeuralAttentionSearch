"""
originally from
https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py
"""
from pathlib import Path
import os
import pickle
import requests
import numpy as np

import tiktoken


def get_shakespeare_dataset(dataset_path: Path, char_level:bool=False, tokenizer:tiktoken.Encoding | None = None):
    dataset_path = dataset_path
    # download the tiny shakespeare dataset
    input_file_path = dataset_path / 'input.txt'
    if not dataset_path.exists():
        os.makedirs(dataset_path, exist_ok=True)
    if not input_file_path.exists():
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    if char_level:
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        print("all the unique characters:", ''.join(chars))
        print(f"vocab size: {vocab_size:,}")
        # get all the unique characters that occur in this text
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        print("all the unique characters:", ''.join(chars))
        print(f"vocab size: {vocab_size:,}")

        # create a mapping from characters to integers
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        def encode(s):
            return [stoi[c] for c in s]  # encoder: take a string, output a list of integers

        def decode(l):
            return ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

        # create the train and test splits
        n = len(data)
        train_data = data[:int(n * 0.9)]
        val_data = data[int(n * 0.9):]

        # encode both to integers
        train_ids = encode(train_data)
        val_ids = encode(val_data)
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile(dataset_path / 'train.bin')
        val_ids.tofile(dataset_path / 'val.bin')

        # save the meta information as well, to help us encode/decode later
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        with open(dataset_path/ 'meta.pkl', 'wb') as f:
            pickle.dump(meta, f)
    else:
        n = len(data)
        train_data = data[:int(n * 0.9)]
        val_data = data[int(n * 0.9):]

        # encode with tiktoken gpt2 bpe
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("gpt2")
        train_ids = tokenizer.encode_ordinary(train_data)
        val_ids = tokenizer.encode_ordinary(val_data)
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile(dataset_path / 'train.bin')
        val_ids.tofile(dataset_path / 'val.bin')

        # train.bin has 301,966 tokens
        # val.bin has 36,059 tokens



