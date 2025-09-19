import datasets
from datasets import load_dataset
from pathlib import Path
import json
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer

output_path = Path.cwd()

# tokneize a sent

def read_potqa(tokenizer, long_bench_dataset_path, output_path, min_length, max_len):
    #http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
    dataset_path = long_bench_dataset_path / 'hotpot_train_v1.1.json'
    with open(dataset_path, 'r') as f:
        raw_dataset = json.load(f)

    all_dataset = []

    random_order = np.random.permutation(len(raw_dataset))
    for idx in random_order:
        dat = raw_dataset[idx]
        new_data = {}
        context = dat['context']
        context_data = []

        for i, ctx in enumerate(context):
            topic = f'{ctx[0]}'
            info = '\n'.join(ctx[1])
            context_data.append(f'Passage {i+1}:\n{topic}\n{info}')

        context = '\n '.join(context_data)

        if len(context) < min_length:
            continue
        num_tokens = len(tokenizer(context, truncation=False, return_tensors="pt").input_ids.flatten())

        if num_tokens > max_len:
            print(f'skip with {num_tokens} tokens')
            continue


        new_data['context'] = context
        new_data['input'] = dat['question']
        new_data['answers'] = dat['answer']

        all_dataset.append(new_data)
        if len(all_dataset) > n_dataset_per_repo:
            break

    with open(output_path / 'hotpotqa.jsonl', 'w') as fl:
        for entry in all_dataset:
            json.dump(entry, fl)
            fl.write('\n')


def read_2wiki(tokenizer, long_bench_dataset_path, output_path, min_length, max_len):
    raw_dataset = load_dataset('xanhho/2WikiMultihopQA', split='train')
    random_order = np.random.permutation(len(raw_dataset))

    all_dataset = []

    for idx in random_order:
        dat = raw_dataset[idx.item()]
        new_data = {}
        titles = dat['context']['title']
        contents = dat['context']['content']
        all_ctx = []

        for i, (title, content) in enumerate(zip(titles, contents)):
            all_ctx.append(f'Passage {i+1}:\n{title}\n{content}')

        context = '\n'.join(all_ctx)

        num_tokens = len(tokenizer(context, truncation=False, return_tensors="pt").input_ids.flatten())

        if num_tokens > max_len:
            print(f'skip with {num_tokens} tokens')
            continue

        new_data['context'] = context
        new_data['input'] = dat['question']
        new_data['answers'] = dat['answer']
        all_dataset.append(new_data)
        if len(all_dataset) > n_dataset_per_repo:
            break

    with open(output_path / '2wikimqa.jsonl', 'w') as fl:
        for entry in all_dataset:
            json.dump(entry, fl)
            fl.write('\n')


def readmusique(tokenizer, long_bench_dataset_path, output_path, min_length, max_len):
    # https://github.com/StonyBrookNLP/musique
    dataset_path = long_bench_dataset_path / 'musique_ans_v1.0_train.jsonl'
    with open(dataset_path) as f:
        raw_dataset = [json.loads(line) for line in f]
    random_order = np.random.permutation(len(raw_dataset))
    all_dataset = []
    for idx in random_order:
        dat = raw_dataset[idx.item()]
        new_data = {}
        contents = dat['paragraphs']
        all_ctx = []

        for i, content in enumerate(contents):
            title = content['title']
            text = content['paragraph_text']
            all_ctx.append(f'Passage {i+1}:\n{title}\n{text}')

        context = '\n'.join(all_ctx)

        num_tokens = len(tokenizer(context, truncation=False, return_tensors="pt").input_ids.flatten())

        if num_tokens > max_len:
            print(f'skip with {num_tokens} tokens')
            continue

        new_data['context'] = context
        new_data['input'] = dat['question']
        new_data['answers'] = dat['answer']

        all_dataset.append(new_data)
        if len(all_dataset) > n_dataset_per_repo:
            break

    with open(output_path / 'musique.jsonl', 'w') as fl:
        for entry in all_dataset:
            json.dump(entry, fl)
            fl.write('\n')


def read_dureader(tokenizer, long_bench_dataset_path, output_path, min_length, max_len):
    # https://github.com/baidu/DuReader/tree/master/DuReader-2.0
    data_path_all = long_bench_dataset_path / 'dureader'
    data_zhidao = data_path_all / 'zhidao.train.json'
    data_search = data_path_all / 'search.train.json'

    n_datasets = 10000
    with open(data_zhidao) as f:
        raw_data_zhidao = [json.loads(line) for i, line in enumerate(f) if i < n_datasets]

    with open(data_search) as f:
        raw_data_search = [json.loads(line) for i, line in enumerate(f) if i < n_datasets]

    len_dataset = len(raw_data_search) + len(raw_data_zhidao)
    random_order = np.random.permutation(len_dataset)
    n_merge_every = 4

    all_datasets = []

    dataset_current = []
    for i, idx in enumerate(random_order):
        if i % n_merge_every == 0:
            dataset_current = []
        dat = raw_data_zhidao[idx] if idx < n_datasets else raw_data_search[idx - n_datasets]
        dataset_current.append(
            {'answers': dat['answers'],
             'input': dat['question'],
             'documents': dat['documents']
             }
        )
        if (i +1) % n_merge_every == 0:
            new_data = {}
            idx_quest = np.random.randint(0, n_merge_every)
            new_data['input'] = dataset_current[idx_quest]['input']
            new_data['answers'] = dataset_current[idx_quest]['answers']
            context_lists = []
            j = 1
            for dataset in dataset_current:
                for doc in dataset['documents']:
                    title_doc = doc['title']
                    pargraph = doc['paragraphs']
                    ctx= f'文章{j}\n标题：{title_doc}\n{pargraph}\n'
                    context_lists.append(ctx)
                    j += 1
            context = '\n'.join(context_lists)
            num_tokens = len(tokenizer(context, truncation=False, return_tensors="pt").input_ids.flatten())

            if num_tokens > max_len:
                print(f'skip with {num_tokens} tokens')
                continue

            new_data['context'] = context
            all_datasets.append(new_data)
            if len(all_datasets) > n_dataset_per_repo:
                break

    with open(output_path / 'dureader.jsonl', 'w') as fl:
        for entry in all_datasets:
            json.dump(entry, fl, ensure_ascii=False)
            fl.write('\n')


def read_narrativeqa(tokenizer, long_bench_dataset_path, output_path, min_length, max_len):
    dataset = load_dataset('deepmind/narrativeqa', split='train')
    random_order = np.random.permutation(len(dataset))

    all_datasets = []
    for i in random_order:
        data = dataset[i.item()]
        new_data = {}
        context = data['document']['text']
        num_tokens = len(tokenizer(context, truncation=False, return_tensors="pt").input_ids.flatten())
        if num_tokens > max_len:
            print(f'skip with {num_tokens} tokens')
            continue

        new_data['context'] = context

        new_data['input'] = data['question']['text']

        all_answers = [dt['text'] for dt in data['answers']]

        answer_id = np.random.randint(len(all_answers))

        new_data['answers'] = all_answers[answer_id]
        all_datasets.append(new_data)
        if len(all_datasets) > n_dataset_per_repo:
            break

    with open(output_path / 'narrativeqa.jsonl', 'w') as fl:
        for entry in all_datasets:
            json.dump(entry, fl)
            fl.write('\n')


def read_qasper(tokenizer, long_bench_dataset_path, output_path, min_length, max_len):
    dataset = load_dataset('allenai/qasper', split='train')

    random_order = np.random.permutation(len(dataset))

    all_datasets = []
    for i in random_order:
        data = dataset[i.item()]
        text = data['full_text']
        section_names = text['section_name']
        paragraphs = text['paragraphs']
        ctx = []
        for sec_name, par in zip(section_names, paragraphs):
            ctx.append(f'{sec_name}\n{par}')
        context = '\n'.join(ctx)
        num_tokens = len(tokenizer(context, truncation=False, return_tensors="pt").input_ids.flatten())
        if len(context) < min_length:
            continue
        if num_tokens > max_len:
            print(f'skip with {num_tokens} tokens')
            continue
        new_data = {}
        new_data['context'] = context
        n_qas = len(data['qas']['question'])
        qas_id = np.random.randint(0, n_qas)
        new_data['input'] =data['qas']['question'][qas_id]
        all_answers = []

        for answer in data['qas']['answers'][qas_id]['answer']:
            if answer['unanswerable']:
                all_answers = 'Unanswerable'
            elif answer['yes_no'] is not None:
                if answer['yes_no']:
                    all_answers = 'Yes'
                else:
                    all_answers = 'No'
            elif answer['free_form_answer'] != '':
                all_answers.append(answer['free_form_answer'])
            else:
                all_answers.append('\n'.join(answer['evidence']))
        if isinstance(all_answers, list):
            new_data['answers'] = '\n'.join(all_answers)
        else:
            new_data['answers'] = all_answers

        all_datasets.append(new_data)
        if len(all_datasets) > n_dataset_per_repo:
            break

    with open(output_path / 'qasper.jsonl', 'w') as fl:
        for entry in all_datasets:
            json.dump(entry, fl)
            fl.write('\n')


def read_gov_repo(tokenizer, long_bench_dataset_path, output_path, min_length, max_len):
    # https://gov-report-data.github.io/
    gao_id_info = long_bench_dataset_path / 'gov-report' / 'split_ids'/ 'gao_train.ids'
    crs_id_info = long_bench_dataset_path / 'gov-report' / 'split_ids'/ 'crs_train.ids'

    with open(gao_id_info, 'r') as f:
        data_gao = f.readlines()

    with open(crs_id_info, 'r') as f:
        data_crs = f.readlines()

    num_gao = len(data_gao)
    num_crs = len(data_crs)
    #len_dataset = num_gao+ num_crs
    len_dataset = num_crs
    random_order = np.random.permutation(len_dataset)
    all_dataset = []
    all_datasets = []
    def dig_unit_no_subsections(input_doc, doc_context=[]):
        if isinstance(input_doc, dict):
            if len(input_doc['subsections']) == 0:
                title = input_doc['section_title']
                paragraph = '\n'.join(input_doc['paragraphs'])
                context = f'{title}\n{paragraph}'
                doc_context.append(context)
            else:
                dig_unit_no_subsections(input_doc['subsections'], doc_context)
        elif isinstance(input_doc, list):
            for doc in input_doc:
                if len(doc['subsections']) == 0:
                    title = doc['section_title']
                    paragraph = '\n'.join(doc['paragraphs'])
                    context = f'{title}\n{paragraph}'
                    doc_context.append(context)
                else:
                    dig_unit_no_subsections(doc['subsections'], doc_context)
        else:
            raise ValueError

    for i in random_order:
        #if i < num_gao:
        if False:
            file_name = data_gao[i].replace('\n', '.json')
            file_name = long_bench_dataset_path / 'gov-report' / 'gao' / file_name
            is_gao = True
        else:
            #file_name = data_crs[i-num_gao].replace('\n', '.json')
            file_name = data_crs[i].replace('\n', '.json')
            file_name = long_bench_dataset_path / 'gov-report' / 'crs' / file_name
            is_gao = False

        with open(file_name) as f:
            raw_dataset = [json.loads(line) for line in f]
        new_data = {}
        title = raw_dataset[0]['title']
        doc_context = [title]
        if is_gao:
            report = raw_dataset[0]['report']
        else:
            report = raw_dataset[0]['reports']
        dig_unit_no_subsections(report, doc_context)
        context = '\n'.join(doc_context)

        if len(context) < min_length:
            continue

        num_tokens = len(tokenizer(context, truncation=False, return_tensors="pt").input_ids.flatten())

        if num_tokens > max_len:
            print(f'skip with {num_tokens} tokens')
            continue

        new_data['context'] = context
        if is_gao:
            if len(raw_dataset[0]['highlight']) > 0:
                highlight_id = np.random.randint(0, len(raw_dataset[0]['highlight']))
            else:
                highlight_id = 0
            highlight = raw_dataset[0]['highlight'][highlight_id]
            new_data['input'] = highlight['section_title']
            new_data['answers'] = highlight['paragraphs']
        else:
            new_data['input'] = ''
            new_data['answers'] = '\n'.join(raw_dataset[0]['summary'])

        all_dataset.append(new_data)
        if len(all_dataset) > n_dataset_per_repo:
            break

    with open(output_path / 'gov_report.jsonl', 'w') as fl:
        for entry in all_dataset:
            json.dump(entry, fl)
            fl.write('\n')


def read_qmsum(tokenizer, long_bench_dataset_path, output_path, min_length, max_len):
    # https://github.com/Yale-LILY/QMSum
    dataset = long_bench_dataset_path / 'QMSum' / 'data' / 'ALL' / 'jsonl'
    data_path = dataset / 'train.jsonl'

    all_dataset = []

    with open(data_path) as f:
        raw_dataset = [json.loads(line) for line in f]
    random_order = np.random.permutation(len(raw_dataset))

    for idx in random_order:
        dat = raw_dataset[idx]
        new_data = {}
        context = []
        for ctx in dat['meeting_transcripts']:
            context.append(f"{ctx['speaker']}:\n{ctx['content']}")
        context = '\n '.join(context)
        num_tokens = len(tokenizer(context, truncation=False, return_tensors="pt").input_ids.flatten())

        if num_tokens > max_len:
            print(f'skip with {num_tokens} tokens')
            continue
        new_data['context'] = context

        qa = dat['general_query_list'][0]
        new_data['input'] = qa['query']
        new_data['answers'] = qa['answer']

        all_dataset.append(new_data)
        if len(all_dataset) > n_dataset_per_repo:
            break

    with open(output_path / 'qmsum.jsonl', 'w') as fl:
        for entry in all_dataset:
            json.dump(entry, fl)
            fl.write('\n')


def read_multi_news(tokenizer, long_bench_dataset_path, output_path, min_length, max_len):
    raw_dataset = load_dataset('alexfabbri/multi_news', split='train')

    random_order = np.random.permutation(len(raw_dataset))
    all_dataset = []
    for idx in random_order:
        data = raw_dataset[idx.item()]
        new_data = {}
        context = data['document']

        if len(context) < min_length:
            continue

        num_tokens = len(tokenizer(context, truncation=False, return_tensors="pt").input_ids.flatten())

        if num_tokens > max_len:
            print(f'skip with {num_tokens} tokens')
            continue

        new_data['input'] = ''
        new_data['context'] = context
        new_data['answers'] = data['summary']

        all_dataset.append(new_data)
        if len(all_dataset) > n_dataset_per_repo:
            break

    with open(output_path / 'multi_news.jsonl', 'w') as fl:
        for entry in all_dataset:
            json.dump(entry, fl)
            fl.write('\n')

def read_vcsum(tokenizer, long_bench_dataset_path, output_path, min_length, max_len):
    # https://github.com/hahahawu/VCSum
    dataset_path = long_bench_dataset_path / 'VCSum' / 'vcsum_data' / 'long_train.txt'
    meeting_path = long_bench_dataset_path / 'VCSum' / 'vcsum_data' / 'overall_context.txt'


    with open(meeting_path) as f:
        meeting_data = [json.loads(line) for line in f]

    with open(dataset_path) as f:
        summary = [json.loads(line) for line in f]

    random_order = np.random.permutation(len(summary))
    all_datasets = []
    for i in random_order:
        summary_data = summary[i]
        id = int(summary_data['id'])

        metting_info = meeting_data[id]
        all_ctx = []
        for speaker_id, metting_ctx in zip(metting_info['speaker'], metting_info['context']):
            ctx = ''.join(metting_ctx)
            all_ctx.append(
                f'讲者{speaker_id}：{ctx}'
            )
        context = '\n'.join(all_ctx)

        num_tokens = len(tokenizer(context, truncation=False, return_tensors="pt").input_ids.flatten())
        if num_tokens > max_len:
            print(f'skip with {num_tokens} tokens')
            continue

        new_data = {}
        new_data['input'] = ''
        new_data['context'] = context
        new_data['answers'] = summary_data['summary']
        all_datasets.append(new_data)

        if len(all_datasets) > n_dataset_per_repo:
            break

    with open(output_path / 'vcsum.jsonl', 'w') as fl:
        for entry in all_datasets:
            json.dump(entry, fl,  ensure_ascii=False)
            fl.write('\n')

def read_triviaQA(tokenizer, long_bench_dataset_path, output_path, min_length, max_len):
    dataset = datasets.load_dataset('mandarjoshi/trivia_qa', 'rc', split='train[:10%]')
    random_order = np.random.permutation(len(dataset))

    all_datasets = []
    n_lower = 2
    n_higher = 6
    n_combined = np.random.randint(n_lower, n_higher)
    i_inner = 1

    all_datasets = []
    dataset_current = []
    for i in random_order:
        dataset_current.append(dataset[i.item()])
        i_inner += 1
        if i_inner >= n_combined:
            new_data = {}
            input_diag = np.random.randint(0, len(dataset_current))
            all_context = []
            for j in range(len(dataset_current)):
                if j != input_diag:
                    search_results = '\n'.join(dataset_current[j]['search_results']['search_context'])
                    question = dataset_current[j]['question']

                    all_answers = dataset_current[j]['answer']['aliases']
                    answer_id = np.random.randint(len(all_answers))
                    answer = all_answers[answer_id]

                    all_context.append(f'Passage:\n: {search_results}\nQuestion:{question}\nAnswer:\n{answer}')

            context = '\n'.join(all_context)

            if len(context) < min_length:
                continue
            num_tokens = len(tokenizer(context, truncation=False, return_tensors="pt").input_ids.flatten())

            if num_tokens > max_len:
                print(f'skip with {num_tokens} tokens')
                print(f'collected dataset: {len(dataset_current)}')
                # we still need to reset these
                i_inner = 1
                n_combined = np.random.randint(n_lower, n_higher)
                dataset_current = []
                continue

            selected_data = dataset_current[input_diag]
            new_data['context'] = context
            all_answers = selected_data['answer']['aliases']
            answer_id = np.random.randint(len(all_answers))
            answer = all_answers[answer_id]
            new_data['answers'] = answer
            search_results = '\n'.join(selected_data['search_results']['search_context'])
            quesiton = selected_data['question']
            new_data['input'] = f'Passage:\n {search_results}\nQuestion:\n{quesiton}\nAnswer:\n'

            all_datasets.append(new_data)

            i_inner = 1
            n_combined = np.random.randint(n_lower, n_higher)
            dataset_current = []

            if len(all_datasets) > n_dataset_per_repo:
                break

    with open(output_path / 'triviaqa.jsonl', 'w') as fl:
        for entry in all_datasets:
            json.dump(entry, fl)
            fl.write('\n')

    import pdb
    pdb.set_trace()

def read_samsum(tokenizer, long_bench_dataset_path, output_path, min_length, max_len):
    dataset = load_dataset('Samsung/samsum', split='train')

    all_dataset = []
    random_order = np.random.permutation(len(dataset))

    n_lower = 10
    n_higher = 50
    n_combined = np.random.randint(n_lower, n_higher)
    i_inner = 0

    all_datasets = []
    dataset_current = []
    for i in random_order:
        dataset_current.append(dataset[i.item()])
        i_inner += 1
        if i_inner >= n_combined:
            new_data = {}
            input_diag = np.random.randint(0, len(dataset_current))
            all_context = []
            for j in range(len(dataset_current)):
                if j != input_diag:
                    dialogue = dataset_current[j]['dialogue']
                    summary = dataset_current[j]['summary']
                    all_context.append(f'Dialogue: {dialogue}\nSummary:{summary}')

            context = '\n'.join(all_context)
            num_tokens = len(tokenizer(context, truncation=False, return_tensors="pt").input_ids.flatten())
            if num_tokens > max_len:
                print(f'skip with {num_tokens} tokens')
                i_inner = 0
                n_combined = np.random.randint(n_lower, n_higher)
                dataset_current = []
                continue

            dialogue_input = dataset_current[input_diag]['dialogue']
            new_data['context'] = context
            new_data['answers'] = dataset_current[input_diag]['summary']
            new_data['input'] = f'Dialogue: {dialogue_input}\nSummary:'

            all_datasets.append(new_data)

            i_inner = 0
            n_combined = np.random.randint(n_lower, n_higher)
            dataset_current = []

            if len(all_datasets) > n_dataset_per_repo:
                break


    with open(output_path / 'samsum.jsonl', 'w') as fl:
        for entry in all_datasets:
            json.dump(entry, fl)
            fl.write('\n')

def read_trec(tokenizer, long_bench_dataset_path, output_path, min_length, max_len):
    dataset = load_dataset("Cogcomp/trec", split='train')
    FINE_LABELS = [
        "Abbreviation",
        "Expression abbreviated",
        "Animal",
        "Organ of body",
        "Color",
        "Invention, book and other creative piece",
        "Currency name",
        "Disease and medicine",
        "Event",
        "Food",
        "Musical instrument",
        "Language",
        "Letter like a-z",
        "Other entity",
        "Plant",
        "Product",
        "Religion",
        "Sport",
        "Element and substance",
        "Symbols and sign",
        "Techniques and method",
        "Equivalent term",
        "Vehicle",
        "Word with a special property",
        "Definition of something",
        "Description of something",
        "Manner of an action",
        "Reason",
        "Group or organization of persons",
        "Individual",
        "Title of a person",
        "Description of a person",
        "City",
        "Country",
        "Mountain",
        "Other location",
        "State",
        "Postcode or other code",
        "Number of something",
        "Date",
        "Distance, linear measure",
        "Price",
        "Order, rank",
        "Other number",
        "Lasting time of something",
        "Percent, fraction",
        "Speed",
        "Temperature",
        "Size, area and volume",
        "Weight",
    ]

    all_dataset = []
    random_order = np.random.permutation(len(dataset))

    n_lower = 10
    n_higher = 100
    n_combined = np.random.randint(n_lower, n_higher)
    i_inner = 0

    all_datasets = []
    dataset_current = []

    i_inner = 0
    for i in random_order:
        dataset_current.append(dataset[i.item()])
        i_inner += 1
        if i_inner >= n_combined:
            new_data = {}
            input_diag = np.random.randint(0, len(dataset_current))
            all_context = []
            for j in range(len(dataset_current)):
                if j != input_diag:
                    question = dataset_current[j]['text']
                    #label = FINE_LABELS[dataset_current[j]['fine_label']].split(':')[-1].capitalize()
                    label = FINE_LABELS[dataset_current[j]['fine_label']]
                    all_context.append(f'Question: {question}\nType: {label}')


            context = '\n'.join(all_context)

            num_tokens = len(tokenizer(context, truncation=False, return_tensors="pt").input_ids.flatten())
            if num_tokens > max_len:
                print(f'skip with {num_tokens} tokens')
                i_inner = 0
                n_combined = np.random.randint(n_lower, n_higher)
                dataset_current = []
                continue

            selected_data = dataset_current[input_diag]

            new_data['context'] = context
            new_input = selected_data['text']
            label = FINE_LABELS[selected_data['fine_label']]
            new_data['answers'] = label
            new_data['input'] = f'Question: {new_input}\Type:'

            all_datasets.append(new_data)

            i_inner = 0
            n_combined = np.random.randint(n_lower, n_higher)
            dataset_current = []

            if len(all_datasets) > n_dataset_per_repo:
                break

    with open(output_path / 'trec.jsonl', 'w') as fl:
        for entry in all_datasets:
            json.dump(entry, fl)
            fl.write('\n')

def read_lcc(tokenizer, long_bench_dataset_path, output_path, min_length, max_len):
    ds_py = load_dataset("microsoft/LCC_python", split='train')
    ds_java = load_dataset("microsoft/LCC_java", split='train')
    ds_csharp = load_dataset("microsoft/LCC_csharp", split='train')

    len_py = len(ds_py)
    len_ja = len(ds_java)
    len_csharp = len(ds_csharp)
    random_order = np.random.permutation(len_py + len_ja + len_csharp)
    all_datasets = []
    for i in random_order:
        if i > len_py + len_ja:
            data = ds_csharp[i.item() - len_py - len_ja]
        elif i > len_py:
            data = ds_java[i.item() - len_py]
        else:
            data = ds_py[i.item()]
        ctx = data['context']
        all_ctx = ctx.rsplit('\n', 2)

        context = all_ctx[0]
        label = all_ctx[1]
        while label.replace(' ', '') == '}':
            all_ctx = context.rsplit('\n', 2)
            context = all_ctx[0]
            label = all_ctx[1]
        context = f'{context}\n'

        num_tokens = len(tokenizer(context, truncation=False, return_tensors="pt").input_ids.flatten())
        if num_tokens > max_len:
            print(f'skip with {num_tokens} tokens')
            continue
        new_data  = {}
        new_data['context'] = context
        new_data['input'] = ''
        new_data['answers']=label

        all_datasets.append(new_data)

        if len(all_datasets) > n_dataset_per_repo:
            break

    with open(output_path / 'lcc.jsonl', 'w') as fl:
        for entry in all_datasets:
            json.dump(entry, fl)
            fl.write('\n')

def read_repobench(tokenizer, long_bench_dataset_path, output_path, min_length, max_len):
    # https://github.com/Leolty/repobench
    data_path_java = long_bench_dataset_path / 'repo-bench' / 'data' / 'train' / 'completion' / 'java' / 'cross_file_first.pkl.gz'
    data_path_py = long_bench_dataset_path / 'repo-bench' / 'data' / 'train' / 'completion' / 'python' / 'cross_file_first.pkl.gz'
    import  gzip
    import pickle

    with gzip.open(data_path_java, 'r') as f:
        data_java = pickle.load(f)['train']

    with gzip.open(data_path_py, 'r') as f:
        data_py = pickle.load(f)['train']

    len_py = len(data_py)
    len_ja = len(data_java)
    random_order = np.random.permutation(len_py + len_ja )
    all_datasets = []
    for i in random_order:
        if i > len_py:
            data = data_java[i-len_ja]
        else:
            data = data_py[i]
        new_data = {}
        context = data['context']

        if len(context) < min_length:
            continue

        num_tokens = len(tokenizer(context, truncation=False, return_tensors="pt").input_ids.flatten())
        if num_tokens > max_len:
            print(f'skip with {num_tokens} tokens')

        new_data['input'] = data['import_statement']
        new_data['context'] = context
        new_data['answers'] = data['next_line']

        all_datasets.append(new_data)

        if len(all_datasets) > n_dataset_per_repo:
            break

    with open(output_path / 'repobench-p.jsonl', 'w') as fl:
        for entry in all_datasets:
            json.dump(entry, fl)
            fl.write('\n')


    import pdb
    pdb.set_trace()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str, default='')
    parser.add_argument('--long_bench_dataset_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='qasper')
    parser.add_argument('--res_dir', type=str, default='')
    parser.add_argument('--seed', type=str, default=0)
    parser.add_argument('--min_len', type=int, default=1000)
    parser.add_argument('--max_len', type=int, default=16000)
    parser.add_argument('--n_dataset_per_repo', type=int, default=300)
    parser.add_argument('--decoding_length', type=int, default=100)

    return parser.parse_args(args)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    args = parse_args()

    seed = args.seed
    np.random.seed(seed)

    n_dataset_per_repo = args.n_dataset_per_repo

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    long_bench_dataset_path = Path(args.long_bench_dataset_path)
    out_path = Path(args.res_dir)

    max_len = args.max_len
    min_length = args.min_length

    all_funcs = dict(
        qmsum=read_qmsum,
        potqa=read_potqa,
        wiki=read_2wiki,
        musique=readmusique,
        duread=read_dureader,
        multi_news=read_multi_news,
        narrativeqa=read_narrativeqa,
        read_qasper=read_qasper,
        gov_repo=read_gov_repo,
        vcsum=read_vcsum,
        triviqa=read_triviaQA,
        samsum=read_samsum,
        trec=read_trec,
        lcc=read_lcc,
        repobench=read_repobench

    )
    all_funcs[args.dataset](
        tokenizer=tokenizer,
        long_bench_dataset_path=long_bench_dataset_path,
        output_path=output_path,
        min_length=min_length,
        max_len=max_len
    )

