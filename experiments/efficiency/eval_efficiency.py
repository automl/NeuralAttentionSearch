import argparse
import pathlib

import torch
from tqdm import tqdm
from pathlib import Path
import glob

import torch.backends.cudnn as cudnn
from transformers import (
    AutoTokenizer,
    GenerationConfig,
)
from concurrent.futures import ThreadPoolExecutor
import os
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, MistralForCausalLM
from copy import copy

from nats.models.architecture import LanguageModels
from nats.models.model_configuration import TransformerArgs
from nats.models.transformer.hf.llama import enable_llama_nats_eval
from nats.models.transformer.hf.llama_triton_flashattn import enable_llama_triton_eval
from nats.models.transformer.hf.mistral import enable_mistral_nats_eval
from nats.models.transformer.hf.lora import config_lora_for_model, get_fine_tune_models
from nats.components.cache.dyn_cache import NAtSCache
os.environ['TRITON_PRINT_AUTOTUNING']='1'
#os.environ['CUDA_LAUNCH_BLOCKING']='1'

def bench_func(func, num_steps=100, num_warmup_steps=5):
    cudnn.benchmark = True
    pbar = tqdm(range(num_warmup_steps), desc="Warming up...")
    for _ in pbar:
        func()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    pbar = tqdm(range(num_steps), desc="Benchmarking Latency and Memory...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in pbar:
        func()
    end.record()
    torch.cuda.synchronize()
    total_time = start.elapsed_time(end)
    avg_time = total_time / num_steps
    print(f"Average latency: {avg_time:.2f} ms")
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Peak memory usage: {peak_memory:.2f} MB")
    return (
        avg_time,
        peak_memory,
    )

def load_model_and_tokenizer(pre_trained_model_path,
                             adapter_path,
                             model_name,
                             adapter_name,
                             seg_att_sliding_window_size: int,
                             use_seg_attn: bool = False,
                             use_triton:bool = False,
                             do_split: bool = True
                             ):
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_path)
    # model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)

    generation_config = GenerationConfig.from_pretrained(pre_trained_model_path)
    eos_token_ids = generation_config.eos_token_id

    #executor = ThreadPoolExecutor(max_workers=32)
    executor = None

    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    if use_seg_attn:
        base_model_name = (pre_trained_model_path).split('/')[-1]
        transformer_args = TransformerArgs(n_msks=8)
        kv_size_info_path = Path(adapter_path) / model_name / adapter_name / 'kv_size.pth'
        #print(kv_size_info_path)

        if kv_size_info_path.exists() and do_split:
            kv_size = torch.load(kv_size_info_path)
            kv_size_order = torch.argsort(kv_size, -1)
        else:
            kv_size_order = None

        transformer_args.local_seq_max_length = seg_att_sliding_window_size
        if 'Llama-3' in base_model_name:
            model = LlamaForCausalLM.from_pretrained(pre_trained_model_path,
                                                     low_cpu_mem_usage=True,
                                                     #attn_implementation = "sdpa",
                                                     attn_implementation= 'flash_attention_2',
                                                     torch_dtype=torch.bfloat16)
            enable_llama_nats_eval(model, transformer_args, executor=executor, )
        elif 'Mistral' in base_model_name:
            model = MistralForCausalLM.from_pretrained(pre_trained_model_path,
                                                       low_cpu_mem_usage=True,
                                                       #attn_implementation='sdpa',
                                                       attn_implementation='flash_attention_2',
                                                       torch_dtype=torch.bfloat16)
            enable_mistral_nats_eval(model, transformer_args, executor=executor, )
        else:
            raise NotImplemented(base_model_name)

        if 'useLora' in adapter_name:
            has_lora_config = False
            involved_modules = []
            for ada_par in adapter_name.split('_'):
                if ada_par.startswith('LORAModule:'):
                    tune_pars = ada_par.replace('LORAModule:', '')
                    #print(f'tune pars:{tune_pars}')
                    has_lora_config = True
                    involved_modules = get_fine_tune_models(tune_pars)
                    break
            if not has_lora_config:
                raise ValueError
            config_lora_for_model(model.model, involved_modules=involved_modules)

        adapter_sd = torch.load(Path(adapter_path) / model_name / adapter_name / 'adapter' / 'adapter_weights.pth')

        for k in adapter_sd.keys():
            print(k)
            assert k in model.model.state_dict()
        model.model.load_state_dict(adapter_sd, strict=False)
    elif use_triton:
        model= LlamaForCausalLM.from_pretrained(pre_trained_model_path,
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.bfloat16,
                                                )

        enable_llama_triton_eval(model)
    else:
        #model = LlamaForCausalLM.from_pretrained(pre_trained_model_path)
        model = LlamaForCausalLM.from_pretrained(pre_trained_model_path,
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.bfloat16,
                                                 #attn_implementation='sdpa'
                                                 attn_implementation='flash_attention_2'
                                                 )
    return model, tokenizer, eos_token_ids


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_dir', type=str, default='')
    parser.add_argument('--dataset_dir', type=str, default='')
    parser.add_argument('--dataset', type=str, default='qasper')
    parser.add_argument('--res_dir', type=str, default='')
    parser.add_argument('--model_name', type=str, )
    parser.add_argument('--use_seg_attn', action='store_true')
    parser.add_argument('--use_triton', action='store_true')
    parser.add_argument('--adapter_path', type=str, )
    parser.add_argument('--adapter_name', type=str, )
    parser.add_argument('--pre_fill_length', type=int, default=5e4)
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--decoding_length', type=int, default=100)
    parser.add_argument('--do_split', action='store_true')

    return parser.parse_args(args)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    args = parse_args()


    #if args.model_path is not None:
    #    assert args.model_name is None
    #    model_name = args.model_path
    #else:
    #    assert args.model_name is not None
    #    model_name = args.model_name
    model_name = args.model_name

    model_path = Path(args.pretrained_model_dir) / model_name
    device = torch.device(f'cuda:{0}')
    use_seg_attn = args.use_seg_attn
    use_triton = args.use_triton
    adapter_path = Path(args.adapter_path) if use_seg_attn else None
    adapter_name = args.adapter_name if use_seg_attn else None


    seg_att_sliding_window_size = None
    if adapter_name is not None:
        adapter_pars = adapter_name.split('_')
        for adp_par in adapter_pars:
            if adp_par.startswith('SWindowlen'):
                seg_att_sliding_window_size = int(adp_par.replace('SWindowlen', ''))

        if seg_att_sliding_window_size is None:
            raise ValueError

    model, tokenizer, eos_token_ids = load_model_and_tokenizer(pre_trained_model_path=str(model_path),
                                                               adapter_path=adapter_path,
                                                               model_name=model_name, adapter_name=adapter_name,
                                                               use_seg_attn=use_seg_attn,
                                                               seg_att_sliding_window_size=seg_att_sliding_window_size,
                                                               use_triton=use_triton,
                                                               do_split=args.do_split
                                                               )
    model = model.to('cuda')
    model.eval()

    context = ""
    pre_fill_length = args.pre_fill_length
    while(len(tokenizer.encode(context)) < pre_fill_length):
        for file in glob.glob("../needle/PaulGrahamEssays/*.txt"):
            with open(file, "r") as f:
                context += f.read()


    input_ids = tokenizer(context, return_tensors="pt").input_ids.to("cuda")[
        :, : pre_fill_length - 1
    ]

    print(input_ids.shape)

    torch.cuda.reset_peak_memory_stats()


    # ctx_latency, ctx_memory = bench_func(func1, num_steps=20, num_warmup_steps=10)
    ctx_latency, ctx_memory = 0, 0

    if use_seg_attn:
        past_key_values = NAtSCache(sliding_window_size=seg_att_sliding_window_size)
    else:
        past_key_values = None
    # warmup triton kernel

    #start_cuda = torch.cuda.Event(enable_timing=True)
    #end_cuda = torch.cuda.Event(enable_timing=True)

    #start_cuda.record()
    #i_start = 200_000
    i_start = 32000
    cumu = 32000

    if input_ids.shape[1] > i_start:
        input_id_ranges = [0, i_start]
        current_len = i_start
        current_len = cumu + current_len
        while(current_len < input_ids.shape[1]):
            print(current_len)
            input_id_ranges.append(current_len)
            current_len = cumu + current_len
        input_id_ranges.append(input_ids.shape[-1])
    else:
        input_id_ranges = [0, input_ids.shape[1]]
    print(input_id_ranges)
    # warmup triton kernels
    if use_seg_attn or use_triton:
        with torch.no_grad():
            for i in range(len(input_id_ranges) - 1):
                start = input_id_ranges[i]
                end = input_id_ranges[i+1]
                print(f'start: {start}')
                print(f'end: {end}')
                outputs = model.model(
                        input_ids=input_ids[:,start:end],
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=False
                        )
                #print(f'seen tokens {past_key_values.get_seq_length()}')
                torch.cuda.empty_cache()
                #torch.cuda.synchronize()
                past_key_values = outputs.past_key_values

                print(
                        f"Peak memory usage in the pre-filling stage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB"
                        )
                torch.cuda.reset_peak_memory_stats()

                #torch.cuda.empty_cache()
            del past_key_values
            del outputs
    if use_seg_attn:
        past_key_values = NAtSCache(sliding_window_size=seg_att_sliding_window_size)
    else:
        past_key_values = None

    start_cuda = torch.cuda.Event(enable_timing=True)
    end_cuda = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_cuda.record()

    with torch.no_grad():
        for i in range(len(input_id_ranges) - 1):
            start = input_id_ranges[i]
            end = input_id_ranges[i+1]
            #print(f'start: {start}')
            #print(f'end: {end}')
            outputs = model.model(
                    input_ids=input_ids[:,start:end],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=False
                    )
            #print(f'seen tokens {past_key_values.get_seq_length()}')
            #torch.cuda.empty_cache()
            #torch.cuda.synchronize()
            past_key_values = outputs.past_key_values

            #print(
            #        f"Peak memory usage in the pre-filling stage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB"
            #        )
            #torch.cuda.reset_peak_memory_stats()

            hidden_states = outputs[0][:, [-1]].clone()
            #torch.cuda.empty_cache()

        logits = model.lm_head(hidden_states)
    end_cuda.record()
    #del hidden_states
    torch.cuda.synchronize()

    time_pre_filling = start_cuda.elapsed_time(end_cuda)/1000
    memory_pre_filling = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f'time used for pre-filling: {time_pre_filling}')
    print(
        f"Peak memory usage in the pre-filling stage: {memory_pre_filling:.2f} MB"
    )

    #past_key_values = outputs.past_key_values
    #print(past_key_values[0][0].shape)
    pred_token_idx = logits.argmax(dim=-1)
    generated_ids = [pred_token_idx.item()]

    def func2():
        with torch.no_grad():
            _ = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )

    gen_latency, gen_memory = bench_func(func2, num_steps=100, num_warmup_steps=10)

    if args.do_split and adapter_name is not None:
        adapter_name = f'{adapter_name}_split'
    adapter_name = adapter_name or 'vanilla'

    output_dir = f'{model_name}/{adapter_name}/{pre_fill_length/1000}/{100}'
    try:
        kv = torch.stack(
                [(cache_model.size_local_kv + cache_model.size_global_kv + getattr(cache_model, 'chunk_fill_size',0)).cpu() / (input_ids.shape[1]) for cache_model in past_key_values.cache_model]
                )
        print(f'kv size: {torch.mean(kv.float())} ')
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "benchmark_result.txt"), "w") as f:
        print(f'Pre-filling time: {time_pre_filling:.2f} s', file=f)
        print(f'Peak Pre-filling memory usage: {memory_pre_filling:.2f} MB', file=f)
        print(f"Average generation time: {gen_latency:.2f} ms", file=f)
        print(f"Peak generation memory usage: {gen_memory:.2f} MB", file=f)
        print(f"Average context time: {ctx_latency:.2f} ms", file=f)
        print(f"Peak context memory usage: {ctx_memory:.2f} MB", file=f)
        print(f"Model name: {model_name}", file=f)
        print(f"Model name: {adapter_name}", file=f)
        print(f"Context length: {args.pre_fill_length}", file=f)




