import os
from pathlib import Path

import transformers.models.llama.modeling_llama
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, MistralForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse

from transformers import (
    AutoTokenizer,
    GenerationConfig,
)

from transformers.modeling_outputs import BaseModelOutputWithPast

from nats.models.model_configuration import TransformerArgs
from nats.models.transformer.hf.llama import enable_llama_nats_eval
from nats.models.transformer.hf.llama_nats_chunk import enable_llama_nats_chunk_eval
from nats.models.transformer.hf.mistral import enable_mistral_nats_eval
from nats.models.transformer.hf.lora import config_lora_for_model, get_fine_tune_models
from nats.components.cache.dyn_cache import NAtSCache

n_msks_for_models = {'Meta-Llama-3.1-8B-Instruct': 8}
model_types = {'Meta-Llama-3.1-8B-Instruct': 'Meta-Llama-3.1-8B-Instruct'}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_dir', type=str, default='')
    parser.add_argument('--dataset_dir', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dureader')
    parser.add_argument('--res_dir', type=str, default='')
    parser.add_argument('--model_name', type=str, default='Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--nats_enable', action='store_true')
    parser.add_argument('--adapter_path', type=str, default='')
    parser.add_argument('--adapter_name', type=str, default='')
    parser.add_argument('--model_use_lora', action='store_true')
    parser.add_argument('--nats_sliding_window_size', type=int, default=16)

    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)


def load_model_and_tokenizer(pre_trained_model_path,
                             adapter_path,
                             model_name,
                             adapter_name,
                             nats_sliding_window_size: int,
                             nats_chunk_size: int,
                             nats_enable: bool = False,
                             ):
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_path)
    # model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)

    generation_config = GenerationConfig.from_pretrained(pre_trained_model_path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    if nats_enable:
        base_model_name = (pre_trained_model_path).split('/')[-1]
        transformer_args = TransformerArgs(chunk_size=nats_chunk_size, local_seq_max_length=nats_sliding_window_size)

        if 'Llama-3' in base_model_name:
            model = LlamaForCausalLM.from_pretrained(pre_trained_model_path,
                                                     low_cpu_mem_usage=True,
                                                     torch_dtype=torch.float16)
            model.model.layers = model.model.layers[:2]
            if nats_chunk_size == 1:
                enable_llama_nats_eval(model, transformer_args)
            else:
                enable_llama_nats_chunk_eval(model, transformer_args)
        elif 'Mistral' in base_model_name:
            model = MistralForCausalLM.from_pretrained(pre_trained_model_path,
                                                       low_cpu_mem_usage=True,
                                                       torch_dtype=torch.bfloat16)
            enable_mistral_nats_eval(model, transformer_args)
        else:
            raise NotImplemented(base_model_name)

        if 'useLora' in adapter_name:
            has_lora_config = False
            involved_modules = []
            for ada_par in adapter_name.split('_'):
                if ada_par.startswith('LORAModule:'):
                    tune_pars = ada_par.replace('LORAModule:', '')
                    print(f'tune pars:{tune_pars}')
                    has_lora_config = True
                    involved_modules = get_fine_tune_models(tune_pars)
                    break
            if not has_lora_config:
                raise ValueError
            config_lora_for_model(model.model, involved_modules=involved_modules)

        adapter_sd = torch.load(Path(adapter_path) / model_name / adapter_name / 'adapter' / 'adapter_weights.pth')
        for k in adapter_sd.keys():
            if k in model.model.state_dict():
                print(k)
            #assert k in model.model.state_dict()
        model.model.load_state_dict(adapter_sd, strict=False)
    else:
        model = LlamaForCausalLM.from_pretrained(pre_trained_model_path)
    return model, tokenizer, eos_token_ids


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "llama-2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    elif "llama-3" in model_name.lower():
        response = (
            response.split(".assistant")[0]
                .split("\n\nQuestion")[0]
                .split("</s>")[0]
                .strip()
        )
    elif "Llama-2-7B-32K-Instruct" in model_name:
        response = (
            response.split("(Document")[0]
                .split("\n\nQuestion")[0]
                .split("\n\nAnswer")[0]
                .split("(Passage")[0]
                .strip()
        )
    return response


def get_pred(
        tokenizer, model, rank, data, max_length, max_gen, prompt_format, dataset, device, model_name, model_path,
        out_path,
        eos_token_ids, nats_enable: bool = False,
        nats_sliding_window_size: int = 16,
        nats_chunk_size: int = 1,
        # used to ensure that this fit triton kernel size TODO Check how to solve this!!!
):
    preds = []
    all_kv_caches = []
    pbar = tqdm(data)
    for idx, json_obj in enumerate(pbar):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc",
                           "repobench-p"]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        simulation_start_idx = input.input_ids.shape[-1]

        pbar.set_description(
            f"Generating for {idx}, len = {input.input_ids.shape[-1]}, prefill {simulation_start_idx}"
        )

        if nats_enable:
            past_key_values = NAtSCache(sliding_window_size=nats_sliding_window_size, chunk_size=nats_chunk_size)
        else:
            past_key_values = None
        with torch.no_grad():
            output = model.model(
                input_ids=input.input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            past_key_values = output.past_key_values

            if isinstance(output, BaseModelOutputWithPast):
                logits = model.lm_head(output[0][:, [-1], :])
                pred_token_idx = logits.argmax(dim=-1)
            else:
                pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            context_length = input.input_ids.shape[-1]
            position_ids = torch.arange(
                context_length, context_length + 1, device=model.device
            ).unsqueeze(0) + 1

            generated_content = [pred_token_idx.item()]
            for _ in range(max_gen - 1):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                    position_ids=position_ids
                )
                position_ids += 1

                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]
                if pred_token_idx.item() in eos_token_ids:
                    break

        context_length = input.input_ids.shape[-1]

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        pred = post_process(pred, model_name)
        answers = json_obj["answers"]
        print(f"Prediction: {pred}")
        print(f"answers: {answers}")

        preds.append(
            {
                "pred": pred,
                "answers": answers,
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )

        print(torch.cuda.max_memory_allocated() / 1024 / 1024)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        if isinstance(past_key_values, NAtSCache):
            kv = []
            kv.append(
                torch.stack(
                    [(cache_model.size_local_kv + cache_model.size_global_kv + getattr(cache_model, 'chunk_fill_size',
                                                                                       0)).cpu() / (
                                 context_length + len(generated_content)) for cache_model
                     in past_key_values.cache_model]
                )
            )
            kv_size = torch.permute(torch.stack(kv), (2, 1, 3, 0))
            print(f'kv size: {torch.mean(kv_size.float())} ')
            all_kv_caches.append(kv_size)

    if len(all_kv_caches) > 0:
        all_kv_caches = torch.stack(all_kv_caches)
        torch.save(all_kv_caches, out_path / 'kv_size.pth', )

    with open(out_path / 'res.jsonl', "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model_name
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    dataset_name = args.dataset
    assert dataset_name in datasets
    if args.e:
        dataset_name_load_name = f'{dataset_name}_e'
        res_dir = Path(args.res_dir) / 'pred_e'
    else:
        dataset_name_load_name = f'{dataset_name}'
        res_dir = Path(args.res_dir) / 'pred'

    dataset_dir = Path(args.dataset_dir)
    data = load_dataset(str(dataset_dir / 'LongBench.py'), dataset_name_load_name, split='test', trust_remote_code=True)
    model_name = args.model_name
    nats_enable = args.nats_enable
    if nats_enable:
        res_dir = res_dir / f'{model_name}' / f'{args.adapter_name}' / dataset_name
    else:
        res_dir = res_dir / model_name / dataset_name

    if not res_dir.exists():
        os.makedirs(res_dir)
    if (res_dir / 'res.json').exists():
        os.remove(res_dir / 'res.json')

    prompt_format = dataset2prompt[dataset_name]
    max_gen = dataset2maxlen[dataset_name]
    data_all = [data_sample for data_sample in data]
    model_path = Path(args.pretrained_model_dir) / model_name

    device = torch.device(f'cuda:{0}')
    nats_enable = args.nats_enable
    adapter_path = Path(args.adapter_path) if nats_enable else None
    adapter_name = args.adapter_name if nats_enable else None

    nats_sliding_window_size = None
    nats_chunk_size = 1
    if adapter_name is not None:
        adapter_pars = adapter_name.split('_')
        for adp_par in adapter_pars:
            if adp_par.startswith('SWindowlen'):
                nats_sliding_window_size = int(adp_par.replace('SWindowlen', ''))
            if adp_par.startswith('chunk'):
                nats_chunk_size = int(adp_par.replace('chunk', ''))

        if nats_sliding_window_size is None:
            raise ValueError
    print(f'eval on {dataset_name} with {model_name} and {adapter_name}')

    model, tokenizer, eos_token_ids = load_model_and_tokenizer(pre_trained_model_path=str(model_path),
                                                               adapter_path=adapter_path,
                                                               model_name=model_name, adapter_name=adapter_name,
                                                               nats_enable=nats_enable,
                                                               nats_chunk_size=nats_chunk_size,
                                                               nats_sliding_window_size=nats_sliding_window_size,
                                                               )
    model = model.to(device)
    model.eval()
    get_pred(tokenizer, model, rank=0, data=data_all, max_length=max_length,
             max_gen=max_gen, prompt_format=prompt_format,
             dataset=dataset_name,
             device=device,
             model_name=model_name, model_path=model_path, out_path=res_dir,
             nats_enable=nats_enable,
             eos_token_ids=eos_token_ids,
             nats_sliding_window_size=nats_sliding_window_size,
             nats_chunk_size=nats_chunk_size,
             )
