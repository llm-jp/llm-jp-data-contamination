import pdb
from utils import obtain_dataset, get_dataset_list
from utils import form_dataset, batched_data_with_indices, clean_dataset, results_caculate_and_draw
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
from tqdm import tqdm
import torch.quantization
import os
import pickle
import numpy as np
import evaluate
import pandas as pd
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer


def levenshtein_distance(str1, str2):
    if len(str1) > len(str2):
        str1, str2 = str2, str1

    distances = range(len(str1) + 1)
    for index2, char2 in enumerate(str2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(str1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances

    return distances[-1]


def strip_code(sample):
    return sample.strip().split('\n\n\n')[0] if '\n\n\n' in sample else sample.strip().split('```')[0]


def truncate_prompt(sample, method_name):
    if f'def {method_name}(' in sample:
        output = sample.replace("'''", '"""')
        output = output[output.find('def ' + method_name):]
        output = output[output.find('"""') + 3:]
        output = output[output.find('"""\n') + 4:] if '"""\n' in output else output[output.find('"""') + 3:]
    else:
        output = sample

    return output


def tokenize_code(sample, tokenizer, length):
    return tokenizer.encode(sample)[:length] if length else tokenizer.encode(sample)


def get_edit_distance_distribution_star(samples, gready_sample, tokenizer, length=100):
    gready_sample = strip_code(gready_sample)
    gs = tokenize_code(gready_sample, tokenizer, length)
    num = []
    max_length = len(gs)
    for sample in samples:
        sample = strip_code(sample)
        s = tokenize_code(sample, tokenizer, length)
        num.append(levenshtein_distance(gs, s))
        max_length = max(max_length, len(s))
    return num, max_length


def calculate_ratio(numbers, alpha=1):
    count = sum(1 for num in numbers if num <= alpha)
    total = len(numbers)
    ratio = count / total if total > 0 else 0
    return ratio


def get_ed(a, b):
    if len(b) == 0:
        return len(a)
    elif len(a) == 0:
        return len(b)
    else:
        dist = np.zeros((len(a) + 1, len(b) + 1))
        for i in range(len(a)):
            for j in range(len(b)):
                if a[i] == b[j]:
                    dist[i + 1, j + 1] = dist[i, j]
                else:
                    dist[i + 1, j + 1] = 1 + min(dist[i, j], dist[i + 1, j], dist[i, j + 1])

        return int(dist[-1, -1])


def get_peak(samples, s_0, alpha):
    lengths = [len(x) for x in samples]
    l = min(lengths)
    l = min(l, 100)
    thresh = int(np.ceil(alpha * l))
    distances = [get_ed(s, s_0) for s in samples]
    rhos = [len([x for x in distances if x == d]) for d in range(0, thresh + 1)]
    #pdb.set_trace()
    peak = sum(rhos)

    return peak
# def bleurt_score(bleurt, predictions, references):
#     """Compute the average BLEURT score over the gpt responses
#
#     Args:
#         predictions (list): List of gpt responses generated by GPT-3.5 under a certain instruction
#         references (list): List of sentence 2 collected from raw dataset (e.g. wnli)
#
#     Returns:
#         bluert_scores: List of bluert scores
#     """
#     bluert_scores = bleurt.compute(predictions=predictions,
#                                    references=references)['scores']
#     return bluert_scores


def rougeL_score(predictions, references):
    """Compute the rougel score over the gpt responses

    Args:
        predictions (list): List of gpt responses generated by GPT-3.5 under a certain instruction
        references (list): List of sentence 2 collected from raw dataset (e.g. wnli)

    Returns:
        rougeL_scores: List of rougeL scores
    """
    rougeL_score = rouge.compute(predictions=predictions,
                                 references=references, use_aggregator=False)['rougeL']
    return rougeL_score

def bleurt_score(bleurt, tokenizer, reference, generations, args):
    bleurt.eval()
    with torch.no_grad():
        inputs = tokenizer([reference for i in range(len(generations))], generations, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        inputs = {key: value.to(args.refer_cuda) for key, value in inputs.items()}
        res = bleurt(**inputs).logits.flatten().tolist()
    return res

def compute_black_box_mia(args):
    dataset_names, length_list = get_dataset_list(args)
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{args.model_size}-deduped",
        revision="step143000",
        cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
        torch_dtype=torch.bfloat16,
        #quantization_config=bnb_config
    ).cuda(args.cuda).eval()
    model = model.to_bettertransformer()
    tokenizer = AutoTokenizer.from_pretrained(
      f"EleutherAI/pythia-{args.model_size}-deduped",
      revision="step143000",
      cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.generation_config.return_dict_in_generate = True
    #bleurt = evaluate.load('bleurt', 'bleurt-20', model_type="metric")
    bleurt_model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20')#.cuda(args.refer_cuda)
    bleurt_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')
    bleurt_model.eval()
    for min_len in length_list:
        args.min_len = min_len
        for dataset_name in dataset_names:
            if os.path.exists(
                    f"{args.save_dir}_{args.dataset_idx}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_ccd_dict.pkl"):
                print(f"{args.dataset_idx} {dataset_name} {args.min_len} {args.model_size} finished")
                continue
            df = pd.DataFrame()
            if args.same_length == True:
                if os.path.isfile(f"{args.save_dir}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_same_length.csv"):
                    df = pd.read_csv(f"{args.save_dir}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_same_length.csv", index_col=0)
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            else:
                if os.path.isfile(f"{args.save_dir}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_all_length.csv"):
                    df = pd.read_csv(f"{args.save_dir}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_all_length.csv", index_col=0)
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            dataset = obtain_dataset(dataset_name, args)
            device = f'cuda:{args.cuda}'
            ccd_dict = {}
            samia_dict = {}
            ccd_dict[dataset_name] = {"member": [], "nonmember": []}
            samia_dict[dataset_name] = {"member": [], "nonmember": []}
            for set_name in ["member", "nonmember"]:
                cleaned_data, orig_indices = clean_dataset(dataset[set_name])
                for idx, (data_batch, orig_indices_batch) in tqdm(enumerate(batched_data_with_indices(cleaned_data, orig_indices, batch_size=args.generation_batch_size))):
                    orig_idx = [item for item in orig_indices_batch]
                    batched_text = [item for item in data_batch]
                    tokenized_inputs = tokenizer(batched_text, return_tensors="pt", truncation=True, padding=True,
                                                 max_length=args.max_length)
                    tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
                    full_decoded = [[] for _ in range(args.generation_batch_size)]
                    input_length = int(min(tokenized_inputs["attention_mask"].sum(dim=1))/2) if (tokenized_inputs["attention_mask"][0].sum() < args.max_input_tokens) else args.max_input_tokens
                    for _ in tqdm(range(args.generation_samples)):
                        if _ == 0:
                            zero_temp_generation = model.generate(input_ids=tokenized_inputs["input_ids"][:, :input_length],
                                                            attention_mask=tokenized_inputs["attention_mask"][:, :input_length],
                                                         temperature=0,
                                                         max_new_tokens=args.max_new_tokens,
                                                        )
                            decoded_sentences = tokenizer.batch_decode(zero_temp_generation["sequences"],
                                                   skip_special_tokens=True)
                            for i in range(zero_temp_generation["sequences"].shape[0]):
                                full_decoded[i].append(decoded_sentences[i])
                        else:
                            generations = model.generate(input_ids=tokenized_inputs["input_ids"][:, :input_length],
                                                         attention_mask=tokenized_inputs["attention_mask"][:, :input_length],
                                                     do_sample=True,
                                                     temperature=args.temperature,
                                                     max_new_tokens=args.max_new_tokens,
                                                     top_k=50,
                                                    )
                            decoded_sentences = tokenizer.batch_decode(generations["sequences"], skip_special_tokens=True)
                            for i in range(zero_temp_generation["sequences"].shape[0]):
                                full_decoded[i].append(decoded_sentences[i])
                            #full_decoded.append(tokenizer.batch_decode(generations["sequences"][:, input_length:], skip_special_tokens=True))
                    bleurt_model = bleurt_model.to(device)
                    for batch_idx in range(zero_temp_generation["sequences"].shape[0]):
                        #peak = get_peak(full_decoded[batch_idx][1:], full_decoded[batch_idx][0], 0.05)
                        dist, ml = get_edit_distance_distribution_star(full_decoded[batch_idx][1:], full_decoded[batch_idx][0],
                                                                       tokenizer, length=1000)
                        #peak = calculate_ratio(dist, 0.05 * ml)
                        bleurt_value = np.array(bleurt_score(bleurt_model, bleurt_tokenizer,  full_decoded[batch_idx][0], full_decoded[batch_idx][1:], args)).mean().item()
                        ccd_dict[dataset_name][set_name].append(sum(dist)/len(dist))
                        samia_dict[dataset_name][set_name].append(bleurt_value)
                    bleurt_model = bleurt_model.cpu()
                    #pdb.set_trace()
            os.makedirs(f"{args.save_dir}_{args.dataset_idx}", exist_ok=True)
            os.makedirs(f"{args.save_dir}_{args.dataset_idx}/{dataset_name}/{args.relative}/{args.truncated}",
                        exist_ok=True)
            pickle.dump(ccd_dict, open(f"{args.save_dir}_{args.dataset_idx}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_ccd_dict.pkl", "wb"))
            pickle.dump(samia_dict, open(f"{args.save_dir}_{args.dataset_idx}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_samia_dict.pkl", "wb"))
            df = results_caculate_and_draw(dataset_name, args, df, method_list=["ccd", "samia"])
            if args.same_length:
                df.to_csv(f"{args.save_dir}_{args.dataset_idx}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_same_length.csv")
            else:
                df.to_csv(f"{args.save_dir}_{args.dataset_idx}/{dataset_name}/{args.relative}/{args.truncated}/{args.min_len}_{args.model_size}_all_length.csv")
