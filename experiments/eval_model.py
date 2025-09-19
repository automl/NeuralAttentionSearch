import os
import random
from pathlib import Path
from functools import partial

import hydra
import numpy as np
import omegaconf
import torch
import wandb
from transformers import AutoTokenizer

from nats.models.architecture import LanguageModels
from nats.models.model_configuration import TransformerArgs, CacheArgs
import datasets
from tqdm import tqdm

from nats.components.cache.dyn_cache import (
    NAtSCache,
)

from seqmodels.utils import check_fp16_dtype


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Otherwise, Conv1D with dilation will be too slow
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


dataset_names = {
    'RedPajama': ['RedPajama2_eval', 'raw_content'],
    'wikitext': ['wikitext_eval', 'text'],
    'pg19': ['pg19_eval', 'text']

}


def tokenize_fn(tokenizer, example, ctx_len: int):
    outputs = tokenizer(
        tokenizer.eos_token.join(example["raw_content"]),
        truncation=False,
        return_tensors="pt",
        pad_to_multiple_of=ctx_len,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, ctx_len)}


baselines = ['cam', 'att_sink', 'h2o']


def eval_model(cfg: omegaconf.DictConfig):
    # Use DDP for multi-gpu training
    seed = int(cfg.seed)
    seed_everything(seed)

    # prepare dataset
    cfg_dataset = cfg.dataset

    if cfg_dataset.name in dataset_names:
        dataset_dir = Path(cfg_dataset.base_dir)
        dataset_name, input_ids = dataset_names[cfg_dataset.name]
        data = datasets.load_from_disk(str(dataset_dir / dataset_name))
    else:
        raise ValueError(f'Unknown dataset name: {cfg_dataset.name}')

    pre_trained_model_path = cfg.model.pre_trained_model_path

    if cfg.model.from_pre_trained:
        pre_trained_model_path = cfg.model.pre_trained_model_path
        tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_path)
        data = np.asarray(tokenizer(tokenizer.eos_token.join(data[input_ids]))[0].ids)
    elif cfg.tokenizer.source == 'tiktoken':
        """
        tokenizer = tiktoken.get_encoding(tokenizer_name)
        column_names = data.column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        tokenized_datasets = data.map(
            partial(process_tiktoken, enc=tokenizer, text_column_name=text_column_name),
            # batched=True,
            num_proc=16,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
        len_dataset = np.sum(tokenized_datasets['len'])
        x = np.asarray(sum(tokenized_datasets['ids'], []))

        filename = dataset_dir / dataset_name / f'test.npy'
        import pdb
        pdb.set_trace()
        np.save(filename, x)

        # """
        filename = dataset_dir / dataset_name / f'test.npy'
        data = np.load(filename)
    else:
        raise ValueError(f'Unknown tokenizer source {cfg.tokenizer.source}')

    eval_size = cfg.eval.eval_size + 1
    batch_size = cfg.eval.batch_size

    n_eval = (len(data) // eval_size) * eval_size
    data = torch.from_numpy(data[:n_eval].reshape(-1, eval_size))
    from torch.nn import CrossEntropyLoss

    path_model = Path(cfg.model.base_dir) / cfg.model.name
    raw_model_path = path_model / 'model'

    model = LanguageModels.load(raw_model_path,)
    record_compress = False
    from seqmodels.models.transformer.transformer import SegmentTransformer
    cache = model.generate_cache(bsz=batch_size, max_seq_len=eval_size)
    if isinstance(cache, NAtSCache):
        record_compress = True

    # model, tokenizer = load(args.model_name_or_path)
    loss_fn = CrossEntropyLoss(reduction="none")

    nlls = torch.zeros([len(data), eval_size - 1])
    device = torch.device('cuda')
    model.device = device
    i_batches = int(np.ceil(len(data) / batch_size))
    print(i_batches)
    i_text = 0

    f16_type = torch.bfloat16 if check_fp16_dtype() == 'bfloat16' else torch.float16

    if record_compress:
        #kv_size = torch.zeros([len(data), eval_size - 1, n_msks])
        kv_size = None
    if cfg.model.type in baselines:

        start_size = cfg.model.start_size
        recent_size = cfg.model.recent_size
        num_hh_tokens = cfg.model.num_hh_tokens
        if cfg.model.type == 'h2o':
            target_path = Path(
                cfg.model.res_dir) / f'{cfg.dataset.name}' / f'{eval_size}' / f'{seed}' / f'{cfg.model.name}_{num_hh_tokens}_{recent_size}' / f'{cfg.model.type}'
        else:
            target_path = Path(
            cfg.model.res_dir) / f'{cfg.dataset.name}' / f'{eval_size}' / f'{seed}' / f'{cfg.model.name}_{start_size}_{recent_size}' / f'{cfg.model.type}'
    else:
        target_path = Path(
            cfg.model.res_dir) / f'{cfg.dataset.name}' / f'{eval_size}' / f'{seed}' / f'{cfg.model.name}' / f'{cfg.model.type}'
    if not target_path.exists():
        os.makedirs(target_path)

    if record_compress:
        kv_size = None
    nlls = torch.zeros([0, eval_size - 1])

    for i_batch in enumerate(range(i_batches)):
        # encodings = tokenizer(text, return_tensors="pt")
        # seq_len = encodings.input_ids.size(1)
        i_text_end = i_text + batch_size
        seq_len = eval_size
        print(f"seq_len: {seq_len}")
        pbar = tqdm(range(0, seq_len - 1))
        text = data[i_text:i_text_end]

        model.eval()
        nll = []
        kv = []
        if cfg.model.type in baselines:
            cache_kwargs = {
                'start_size': start_size,
                'recent_size': recent_size,
                'num_hh_tokens': num_hh_tokens
            }

            cache = model.generate_cache(bsz=len(text), max_seq_len=seq_len * 2, cache_type=cfg.model.type,
                                         cache_init_kwargs=cache_kwargs)
        else:
            cache = model.generate_cache(bsz=len(text), max_seq_len=seq_len * 2)
        for idx in pbar:
            input_ids = text[:, idx: idx + 1].long().to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True, dtype=f16_type):
                    outputs = model(
                        input_ids,
                        start_pos=idx,
                        cache=cache
                    )[0]
                    logits = outputs.view(-1, outputs.size(-1))
                    label = text[:, idx + 1: idx + 2].long().to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
            nll.append(neg_log_likelihood.cpu())
            if record_compress:
                kv.append(
                    torch.stack(
                        [(cache_model.size_local_kv+cache_model.size_global_kv) for cache_model in cache.cache_model]
                    )
                )

        nll = torch.vstack(nll).T
        nlls = torch.cat([nlls, nll])
        i_text = i_text_end


        ppl = torch.exp(nll.mean())
        print(f'ppl at sequence {i_batch}: {ppl}')


        if record_compress:
            if kv_size is None:
                kv_size = torch.permute(torch.stack(kv), (2, 1, 3, 0))
            else:
                kv_size = torch.cat([kv_size, torch.permute(torch.stack(kv), (2, 1, 3, 0))])

            #torch.save(kv_size, target_path / f'KV_size.pth')
        i_text = i_text_end

    ppl = torch.exp(nlls.mean())
    print(ppl)

    torch.save(nlls, target_path / f'nll.pth')
    if record_compress:
        kv_size = kv_size.mean(0)
        torch.save(kv_size, target_path / f'KV_size.pth')


@hydra.main(config_path="configs", config_name="eval.yaml")
def main(cfg: omegaconf.DictConfig):
    # Multi-gpu training
    if cfg.n_gpus > 1:
        eval_model(cfg)
    else:
        with omegaconf.open_dict(cfg):
            cfg.local_rank = 0
            cfg.world_rank = 0
        eval_model(cfg)


if __name__ == '__main__':
    main()
