from typing import Callable

from functools import partial
from datasets import Dataset
import omegaconf
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import transformers
from transformers import (
    AutoTokenizer,
)
import tiktoken

from nats.models.tokenizer import LLAMA3Tokenizer

from nats.training.data.shakespeare import get_shakespeare_dataset
from nats.training.data.redpajama import gopher_rules_pass

DATASET_KEYS = {
    'openwebtext': 'openwebtext',
    'RedPajama2': 'togethercomputer/RedPajama-Data-V2',
    'pg19': 'deepmind/pg19',
    'arxiv_math':'hoskinson-center/proof-pile'
}


def get_raw_datasets(dataset_cfg: omegaconf.DictConfig):
    dataset_source = dataset_cfg.get('source', 'customized')
    if dataset_source == 'HF':
        dataset_name = DATASET_KEYS[dataset_cfg.name]
        raw_datasets = load_dataset(
            dataset_name,
            dataset_cfg.get('config_name', None),
            # dataset_cfg.config_name,
            cache_dir=dataset_cfg.cache_dir,
            streaming=dataset_cfg.streaming,
        )
        if dataset_cfg.name == 'openwebtext':
            # https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
            # owt by default only contains the 'train' split, so create a test split
            split_dataset = raw_datasets["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
            split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val
            return split_dataset

        if dataset_cfg.name == 'RedPajama2':
            raw_datasets['train'] = list(filter(gopher_rules_pass, raw_datasets["train"]))

        if "val" not in raw_datasets.keys():
            validation_split_percentage = dataset_cfg.get('validation_split_percentage', 5)
            raw_datasets["val"] = load_dataset(
                dataset_name,
                dataset_cfg.get('config_name', None),
                split=f"train[:{validation_split_percentage}%]",
                cache_dir=dataset_cfg.cache_dir,
                streaming=dataset_cfg.streaming,
            )
            raw_datasets["train"] = load_dataset(
                dataset_name,
                dataset_cfg.get('config_name', None),
                split=f"train[{validation_split_percentage}%:]",
                cache_dir=dataset_cfg.cache_dir,
                streaming=dataset_cfg.streaming,
            )
    else:
        # TODO initialize a dataset here!
        raise NotImplementedError

    return raw_datasets


def prepare_dataset(cfg: omegaconf.DictConfig, tokenizer=None):
    # prepare tokenizer
    if tokenizer is None:
        if cfg.dataset.char_level:
            tokenizer = None
        else:
            tokenizer_name = cfg.tokenizer.model_name
            if cfg.tokenizer.source == 'HF':
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=tokenizer_name
                )
            elif cfg.tokenizer.source == 'tiktoken':
                tokenizer = tiktoken.get_encoding(tokenizer_name)
            elif cfg.tokenizer.source == 'llama3':
                tokenizer = LLAMA3Tokenizer(tokenizer_name)
            else:
                raise ValueError(f'Unknown tokenizer source {cfg.tokenizer.source}')

    # prepare dataset
    cfg_dataset = cfg.dataset

    if cfg_dataset.char_level:
        dataset_dir = Path(cfg_dataset.base_dir) / f'{cfg_dataset.name}_charlevel'
    else:
        dataset_dir = Path(cfg_dataset.base_dir) / cfg_dataset.name

    if not (dataset_dir / 'train.bin').exists():
        if cfg_dataset.name == 'shakespeare':
            if not cfg_dataset.char_level:
                assert hasattr(tokenizer, 'encode_ordinary')
            get_shakespeare_dataset(
                Path(dataset_dir),
                char_level=cfg_dataset.char_level,
                tokenizer=tokenizer
            )
        else:
            raw_datasets = get_raw_datasets(cfg_dataset, )
            tokenize_dataset(raw_datasets, cfg_dataset, partial(process_tiktoken, enc=tokenizer))



class TiktokenProcess:
    def __init__(self, enc, column_names):
        self.enc = enc
        self.text_column_name = column_names

    def __call__(self, example):
        ids = self.enc.encode_ordinary(example[self.text_column_name])  # encode_ordinary ignores any special tokens
        ids.append(self.enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out


class HFProcess:
    def __init__(self, enc, column_names):
        self.enc = enc
        self.text_column_name = column_names

    def __call__(self, example):
        return self.enc(example[self.text_column_name])


def process_tiktoken(example, text_column_name, enc,):
    ids = enc.encode_ordinary(example[text_column_name])  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out


def process_hf(examples, text_column_name, tokenizer):
    return tokenizer(examples[text_column_name])


def tokenize_dataset(raw_datasets: Dataset,
                     dataset_cfg: omegaconf.DictConfig,
                     tokenize_func: Callable):
    """
    This function is mainly inspired by: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
    """

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    tokenized_datasets = raw_datasets.map(
        partial(tokenize_func, text_column_name=text_column_name),
        # batched=True,
        num_proc=16,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    dataset_dir = Path(dataset_cfg.base_dir) / dataset_cfg.name
    if not dataset_dir.exists():
        os.makedirs(dataset_dir, exist_ok=True)

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized_datasets.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = dataset_dir / f'{split}.bin'
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
