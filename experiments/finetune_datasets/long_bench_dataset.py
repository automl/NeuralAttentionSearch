import torch
from datasets import load_dataset
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Dict

import numpy as np
import torch
import transformers
import json
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
from torchtune.datasets import ConcatDataset


def get_dataset(dataset_name, dataset_path=None,split="train", size=None):
    dataset_path = dataset_path or 'json'
    dataset = load_dataset(dataset_path, data_files=dataset_name, split=split)
    if size is not None:
        dataset = dataset.select(range(size))
    return dataset


def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


class LongBenchSingleDataset(Dataset):
    def __init__(self, dataset_path, dataset_name, tokenizer, prompt_format, max_length: int=20000, pad_to_multiple_of=128, decoding_simulation_length:int=0,
                 with_prompt:bool=True,
                 ):
        with open(dataset_path / f'{dataset_name}.jsonl') as f:
             dataset = [json.loads(line) for line in f]
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.prompt_format = prompt_format
        self.prompt_start = tokenizer(prompt_format.split('{context}')[0], truncation=False, return_tensors="pt").input_ids[0]
        self.max_length=max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.tokenizer = tokenizer
        self.decoding_simulation_length=decoding_simulation_length
        self.with_prompt = with_prompt
        print(f'dataset {dataset_name} max_length:{self.max_length}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data = self.dataset[i]
        if self.with_prompt:
            prompt = self.prompt_format.format(**data)
            input_ids = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if 'answer' in data:
                answers = self.tokenizer(', '.join(data['answer']), truncation=False, return_tensors="pt").input_ids[0]
            elif 'answers' in data:
                answers = self.tokenizer(', '.join(data['answers']), truncation=False, return_tensors="pt").input_ids[0]
            else:
                print(data.keys())
                raise ValueError
            answers = torch.cat([answers, torch.tensor([self.tokenizer.eos_token_id])])
            len_data = len(answers) + len(input_ids)
            if len_data > self.max_length:
                for _ in range(5):
                    new_idx = np.random.randint(len(self.dataset))
                    data = self.dataset[new_idx]
                    prompt = self.prompt_format.format(**data)
                    input_ids = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
                    if 'answer' in data:
                        answers = self.tokenizer(', '.join(data['answer']), truncation=False, return_tensors="pt").input_ids[0]
                    elif 'answers' in data:
                        answers = self.tokenizer(', '.join(data['answers']), truncation=False, return_tensors="pt").input_ids[0]
                    else:
                        print(data.keys())
                        raise ValueError
                    len_data = len(answers) + len(input_ids)
                    if len_data < self.max_length:
                        break
            if len_data > self.max_length:
                n_removed = len_data - self.max_length
                print(f'for dataset {self.dataset_name}, a sequence is too long with length {len_data}')
                # we preserve the first few part of the context
                n_removed = len(self.prompt_start ) + n_removed
                input_ids = torch.cat([self.prompt_start, input_ids[n_removed:]])

            len_data = len(answers) + len(input_ids)
            len_label = len(answers) + self.decoding_simulation_length
            label_start = max(len_data - len_label, 0)
            label_end = len_data
            if len_data % self.pad_to_multiple_of != 0:
                padding_length = self.pad_to_multiple_of - (len_data % self.pad_to_multiple_of)
                padding_value = torch.full([padding_length], fill_value=self.tokenizer.eos_token_id, dtype=input_ids.dtype, device=input_ids.device)
                data = torch.cat([input_ids, answers, padding_value])
            else:
                data = torch.cat([input_ids, answers])
            labels = torch.ones_like(data) * -1

            #print(f'len id: {input_ids.shape}')
            #print(f'len answers: {answers.shape}')
            #print(f'lable start: {label_start}')
            #print(f'label end: {label_end}')

            labels[label_start:label_end] = 1
            labels[label_end:] = -100
            return dict(input_ids=data, labels=labels)
        else:
            input_ids = self.tokenizer(data['context'], truncation=False, return_tensors="pt").input_ids[0]
            len_data = len(input_ids)
            if len_data > self.max_length:
                for _ in range(5):
                    new_idx = np.random.randint(len(self.dataset))
                    data = self.dataset[new_idx]
                    input_ids = self.tokenizer(data['context'], truncation=False, return_tensors="pt").input_ids[0]
                    len_data = len(input_ids)
                    if len_data < self.max_length:
                        break
            if len_data > self.max_length:
                input_ids = input_ids[-self.max_length:]
            len_data = len(input_ids)
            if len_data % self.pad_to_multiple_of != 0:
                padding_length = self.pad_to_multiple_of - (len_data % self.pad_to_multiple_of)
                padding_value = torch.full([padding_length], fill_value=self.tokenizer.eos_token_id,
                                           dtype=input_ids.dtype, device=input_ids.device)
                data = torch.cat([input_ids, padding_value])
            else:
                data = input_ids

            labels = torch.ones_like(data)
            labels[len(input_ids):] = -100
            return dict(input_ids=data, labels=labels)


from torch.utils.data import Dataset, IterableDataset, ConcatDataset
from experiments.finetune_datasets.duo_attn_dataset import MultiplePasskeyRetrievalDataset

prompt_format = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}


class LongBenchDataset(ConcatDataset):
    def __init__(self,
                 dataset_path: Path,
                 tokenizer: transformers.AutoTokenizer,
                 available_dataset = [
                     #'musique', 'qmsum', 'vcsum', 'lsht', 'dureader', 'narrativeqa', 'multifieldqa_zh', 'passage_retrieval_zh'
                     '2wikimqa', 'dureader', 'gov_report', 'hotpotqa', 'lcc', 'multi_news', 'musique', 'narrativeqa', 'qasper', 'qmsum', 'repobench-p', 'samsum', 'trec', 'triviaqa', 'vcsum',
                 ],
                 max_length: int= 20000,
                 pad_to_multiple_of=128,
                 n_datas = 7000,
                 decoding_simulation_length:int = 0,
                 with_prompt:bool=False,
                 cfg_dataset=None
                 ):
        all_datasets = []
        n_longbench_dataset = 0
        for dataset_name in available_dataset:
            dataset = LongBenchSingleDataset(dataset_path, dataset_name, tokenizer,
                                             prompt_format=prompt_format[dataset_name],
                                             max_length=max_length,
                                             pad_to_multiple_of=pad_to_multiple_of,
                                             decoding_simulation_length=decoding_simulation_length,
                                             with_prompt=with_prompt)
            n_longbench_dataset += len(dataset)
            all_datasets.append(dataset)
        if n_longbench_dataset < n_datas:
            assert cfg_dataset is not None
            print(f'load {cfg_dataset.name} from {cfg_dataset.dataset_path}')
            haystack_dataset = get_dataset(cfg_dataset.name, cfg_dataset.dataset_path, split="train")
            dataset_retrieve = MultiplePasskeyRetrievalDataset(
                haystack_dataset,
                tokenizer,
                max_length=cfg_dataset.context_length_max,
                min_depth_ratio=cfg_dataset.min_needle_depth_ratio,
                max_depth_ratio=cfg_dataset.max_needle_depth_ratio,
                context_length_min=cfg_dataset.context_length_min,
                context_length_max=cfg_dataset.context_length_max,
                context_lengths_num_intervals=cfg_dataset.context_lengths_num_intervals,
                depth_ratio_num_intervals=cfg_dataset.depth_ratio_num_intervals,
                num_passkeys=cfg_dataset.num_passkeys,
                pad_to_multiple_of=128,
                num_datas=n_datas-n_longbench_dataset
            )
            all_datasets.append(dataset_retrieve)
        super().__init__(all_datasets)


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        ret_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        for key in instances[0].keys():
            if key not in ret_dict:
                ret_dict[key] = torch.stack([instance[key] for instance in instances])
        return ret_dict


def get_supervised_dataloader(
    dataset, tokenizer, batch_size, num_workers=4, shuffle=True, sampler=None
):
    collator = DataCollator(tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        shuffle=None if sampler is not None else shuffle,
        sampler=sampler,
    )
    return dataloader

