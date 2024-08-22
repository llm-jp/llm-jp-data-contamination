import json

import datasets
import pandas as pd
from eval import create_random_samples
import torch
from collections import defaultdict
from sklearn.metrics import auc, roc_curve
import zlib
from torch.nn import CrossEntropyLoss
from scipy.stats import entropy, ks_2samp, kurtosis, wasserstein_distance
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict, Dataset
from itertools import islice
import re
from tqdm import tqdm
import pickle
import random
import seaborn as sns
import os
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast, GradScaler


def batched_data_with_indices(data_list, indices_list, batch_size):
    data_iter = iter(data_list)
    indices_iter = iter(indices_list)
    while True:
        data_batch = list(islice(data_iter, batch_size))
        indices_batch = list(islice(indices_iter, batch_size))
        if not data_batch:
            break
        yield (data_batch, indices_batch)


def clean_dataset(dataset):
    invalid_pattern = re.compile(r'^\s*$')

    def is_valid(text):
        return not invalid_pattern.match(text) and len(text) > 5

    cleaned_data = []
    orig_indices = []

    for idx, data in enumerate(dataset):
        if is_valid(data):
            cleaned_data.append(data)
            orig_indices.append(idx)

    return cleaned_data, orig_indices

def form_dataset(dataset_name, args):
    if "WikiMIA" in dataset_name:
        dataset_lengths = ["32", "64", "128", "256"] if "all" in dataset_name else [sub_string for sub_string in
                                                                                    ["32", "64", "128", "256"] if
                                                                                    sub_string in dataset_name]
        mia_dataset = None
        for length in dataset_lengths:
            # Load the dataset for the specific length
            dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{length}")
            member_data = dataset.filter(lambda example: example['label'] == 1)["input"]
            non_member_data = dataset.filter(lambda example: example['label'] == 0)["input"]
            # If mia_dataset does not exist, initialize it with the first loaded dataset
            if mia_dataset is None:
                mia_dataset = DatasetDict({
                    'member': member_data,
                    'nonmember': non_member_data
                })
            else:
                mia_dataset["member"].extend(member_data)
                mia_dataset["nonmember"].extend(non_member_data)
        return mia_dataset
    else:
        if args.relative == "relative":
            dataset = datasets.load_from_disk(f"./{args.load_dir}_relative/{dataset_name}/{args.min_len}")
            member = random.sample(dataset['member']['data'], min(args.samples, len(dataset['member']['data'])))
            nonmember = random.sample(dataset['nonmember']['data'], min(args.samples, len(dataset['nonmember']['data'])))
            dataset = DatasetDict({
                'member': member,
                'nonmember': nonmember
            })
        else:
            min_len = args.min_len if args.min_len != 0 else 5
            max_len = 100 if args.min_len == 0 else min_len + 100
            dataset = datasets.load_from_disk(f"./{args.load_dir}_{args.truncated}/{dataset_name}/{min_len}_{max_len}")
            member = random.sample(dataset['member']['data'], min(args.samples, len(dataset['member']['data'])))
            if args.same_length:
                nonmember = random.sample(dataset['nonmember']['data'], min(args.samples, len(dataset['nonmember']['data'])))
            else:
                nonmember = random.sample(dataset['full_nonmember']['data'], min(args.samples, len(dataset['full_nonmember']['data'])))
            dataset = DatasetDict({
                'member': member,
                'nonmember': nonmember
            })
    return dataset


def caculate_outputs(model, tokenizer, text_batch, device, min_len=50):
    tokenized_inputs = tokenizer(text_batch,
                                 return_tensors="pt",
                                 truncation=True,
                                 padding=True,
                                 max_length=min_len+100,
                                 )
    tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
    target_labels = tokenized_inputs["input_ids"].clone().to(device)
    target_labels[tokenized_inputs["attention_mask"] == 0] = -100
    outputs = model(**tokenized_inputs, labels=target_labels)
    # grad_norms = []
    # for param in model.parameters():
    #     if param.grad is not None:
    #         grad_norms.append(param.grad.detach().norm(2))
    #         pdb.set_trace()
    # grad_norm = torch.stack(grad_norms).mean()
    return outputs, tokenized_inputs, target_labels#, grad_norm




def mix_distribution(dict, dataset_name, title, args, ratio=0.8, total_num=10000):
    train_data = dict[dataset_name]["member"]
    test_data = dict[dataset_name]["nonmember"]
    train_data_num = total_num*ratio
    test_data_num = total_num*(1-ratio)
    train_data = random.sample(train_data, min(int(train_data_num), len(train_data)))
    test_data = random.sample(test_data, min(int(test_data_num), len(test_data)))
    combined_data = train_data + test_data
    # 画分布图
    plt.figure(figsize=(10, 5))
    sns.kdeplot(combined_data, label='Mixed Distribution', alpha=1, shade=True)
    sns.kdeplot(train_data, label='Train Distribution', alpha=1, shade=True)
    sns.kdeplot(test_data, label='Test Distribution', alpha=1, shade=True)
    # 设置标题和轴标签
    plt.title(f'Data Distribution of mixed distribution at ratio {ratio} for {dataset_name} at {args.model_size} model')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    os.makedirs(f"mixed_figure/{dataset_name}", exist_ok=True)
    plt.savefig(f'mixed_figure/{dataset_name}/{title} histogram at {args.model_size} model at ratio {ratio}.png')
    # 显示图表
    plt.show()
    return train_data, test_data, combined_data

def figure_draw(data_dict, title,dataset_name, args):
    plt.figure(figsize=(10, 5))
    fig, axs = plt.subplots(len(data_dict), figsize=(10, 5 * len(data_dict)))
    axs = np.atleast_2d(axs)
    for ax, (dataset_name, dataset_loss) in zip(axs.flatten(), data_dict.items()):
        for phase_name, phase_loss in dataset_loss.items():
            #weights = np.ones_like(phase_loss) / len(phase_loss)
            #ax.hist(phase_loss, bins=100, label=phase_name, alpha=0.5, weights=weights)
            sns.kdeplot(phase_loss, ax=ax, label=phase_name, alpha=0.5, bw_adjust=0.5, shade=True)
        ax.set_title(f'{dataset_name} {title} histogram  at {args.model_size} model')
        ax.set_xlabel(title)
        ax.set_ylabel('Density')
        ax.legend()
    plt.tight_layout()
    os.makedirs(f"{args.save_dir}_figures/{dataset_name}/{args.truncated}/{args.min_len}_{args.min_len+100}", exist_ok=True)
    if args.same_length == True:
        plt.savefig(f"{args.save_dir}_figures/{dataset_name}/{args.truncated}/{args.min_len}_{args.min_len+100}/{title}_histograms_{args.model_size}_{dataset_name}_same_len.png")
    else:
        plt.savefig(f"{args.save_dir}_figures/{dataset_name}/{args.truncated}/{args.min_len}_{args.min_len+100}/{title}_histograms_{args.model_size}_{dataset_name}_all_len.png")
    plt.show()




def caculate_loss_instance(idx, logits, target_labels):
    logits_i = logits[idx].unsqueeze(0)  # Shape (1, seq_length, vocab_size)
    target_i = target_labels[idx].unsqueeze(0)  # Shape (1, seq_length)
    shift_logits = logits_i[:, :-1, :].contiguous()
    shift_labels = target_i[:, 1:].contiguous()
    # 计算交叉熵损失并移除填充 token 贡献
    loss_i = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                             )
    # Create a mask to ignore the loss from padding tokens
    valid_mask = shift_labels != -100
    # 只有有效的 token 计算损失
    loss_i = loss_i * valid_mask.view(-1)
    # 计算每个样本的平均损失
    loss_i = loss_i.sum() / valid_mask.sum()
    return loss_i


def feature_collection(model, tokenizer, dataset, args, dataset_name, min_len=50, upper_limit=10000, refer_model=None, refer_tokenizer=None):
    device = f'cuda:{args.cuda}'
    refer_device = f'cuda:{args.refer_cuda}'
    loss_collect = []
    mink_collect = []
    mink_plus_collect = []
    ppl_collect = []
    zlib_collect = []
    ref_loss_collect = []
    grad_collect = []
    idx_list = []
    cleaned_data, orig_indices = clean_dataset(dataset)
    for idx, (data_batch, orig_indices_batch) in tqdm(
            enumerate(batched_data_with_indices(cleaned_data, orig_indices, batch_size=args.batch_size))):
        orig_idx = [item for item in orig_indices_batch]
        batched_text = [item for item in data_batch]
        outputs, tokenized_inputs, target_labels = caculate_outputs(model, tokenizer, batched_text, device=device, min_len=min_len)
        refer_outputs, refer_tokenized_inputs, refer_target_labels = caculate_outputs(refer_model, refer_tokenizer, batched_text, device=refer_device, min_len=min_len)
        batch_mink_plus_avg, batch_mink_avg = calculate_mink_and_mink_plus(outputs[1], tokenized_inputs)
        loss_value_list, ppl_value_list, zlib_value_list, grad_value_list = caculate_instance_loss_perplexity_zlib(outputs[1], target_labels, batched_text, model, tokenized_inputs, tokenizer)
        mink_plus_collect.extend(batch_mink_plus_avg)
        mink_collect.extend(batch_mink_avg)
        loss_collect.extend(loss_value_list)
        ppl_collect.extend(ppl_value_list)
        zlib_collect.extend(zlib_value_list)
        grad_collect.extend(grad_value_list)
        idx_list.extend(orig_idx)
        if refer_model is not None:
            ref_loss, ref_logits = refer_outputs[:2]
            ref_log_probabilities = torch.nn.functional.log_softmax(ref_logits, dim=-1)
            ref_probabilities = torch.nn.functional.softmax(ref_logits, dim=-1)
            refer_loss_value_list, _, _, _ = caculate_instance_loss_perplexity_zlib(refer_outputs[1], refer_target_labels, batched_text, refer_model, refer_tokenized_inputs, refer_tokenizer)
        ref_loss_collect.extend(refer_loss_value_list)
    return loss_collect, mink_collect, ppl_collect, mink_plus_collect, zlib_collect, ref_loss_collect, idx_list, grad_collect

def calculate_mean_var(dict, dataset_name, split_set):
    for idx1, set1 in enumerate(split_set):
        values = np.array(dict[dataset_name][set1])
        values = values[np.isnan(values)==False]
        mean = np.mean(values)
        var = np.var(values)
        std = np.std(values)
        kur = kurtosis(values)
        print("The mean, variance, std and kurtosis of {} in {} set are {},  {}, {} and {}".format(dataset_name, set1, mean, var, std, kur))
    return mean, var
def js_divergence(dict, dataset_name, split_set = ["train", "valid", "test"]):
    # Ensure p and q sum to 1
    js_matrix = np.zeros((len(split_set), len(split_set)))
    for idx1, set1 in enumerate(split_set):
        for idx2, set2 in enumerate(split_set):
            values = np.array(dict[dataset_name][set1])
            values1 = values[np.isnan(values) == False]
            values = np.array(dict[dataset_name][set2])
            values2 = values[np.isnan(values) == False]
            hist1, bin_edges = np.histogram(values1, bins=300, density=True)
            hist2, _ = np.histogram(values2, bins=300, density=True)
            eps = 1e-10
            hist1 += eps
            hist2 += eps
            # 确保向量总和为1（归一化），表示概率分布
            hist1 /= hist1.sum()
            hist2 /= hist2.sum()
            m = 0.5 * (hist1 + hist2)
            js_matrix[idx1][idx2] = 0.5 * entropy(hist1, m) + 0.5 * entropy(hist2, m)
    return js_matrix#close to zero means the two distributions are similar

def ks_hypothesis(dict, dataset_name, split_set = ["train", "valid", "test"]):
    ks_statistic_matrix = np.zeros((len(split_set), len(split_set)))
    ks_p_value_matrix = np.zeros((len(split_set), len(split_set)))
    for idx1, set1 in enumerate(split_set):
        for idx2, set2 in enumerate(split_set):
            values = np.array(dict[dataset_name][set1])
            values1 = values[np.isnan(values) == False]
            values = np.array(dict[dataset_name][set2])
            values2 = values[np.isnan(values) == False]
            ks_stat, p_value = ks_2samp(values1, values2)
            ks_statistic_matrix[idx1][idx2] = ks_stat
            ks_p_value_matrix[idx1][idx2] = p_value
    return ks_statistic_matrix, ks_p_value_matrix#close to zero means the two distributions are similar

def wasserstein_distance_caculate(dict, dataset_name,  split_set = ["train", "valid", "test"]):
    ws_matrix = np.zeros((len(split_set), len(split_set)))
    for idx1, set1 in enumerate(split_set):
        for idx2, set2 in enumerate(split_set):
            values = np.array(dict[dataset_name][set1])
            values1 = values[np.isnan(values) == False]
            values = np.array(dict[dataset_name][set2])
            values2 = values[np.isnan(values) == False]
            ws_stat = wasserstein_distance(values1, values2)
            ws_matrix[idx1][idx2] = ws_stat
    return ws_matrix



def results_caculate_and_draw(dataset_name, args, df, method_list):
    split_set = ["member", "nonmember"]
    # loss_dict = pickle.load(open(f"{args.dir}/{dataset_name}/{args.truncated}/{args.min_len}_{args.model_size}_loss_dict.pkl", "rb"))
    # prob_dict = pickle.load(open(f"{args.dir}/{dataset_name}/{args.truncated}/{args.min_len}_{args.model_size}_prob_dict.pkl", "rb"))
    # ppl_dict = pickle.load(open(f"{args.dir}/{dataset_name}/{args.truncated}/{args.min_len}_{args.model_size}_ppl_dict.pkl", "rb"))
    # mink_plus_dict = pickle.load(open(f"{args.dir}/{dataset_name}/{args.truncated}/{args.min_len}_{args.model_size}_mink_plus_dict.pkl", "rb"))
    # zlib_dict = pickle.load(open(f"{args.dir}/{dataset_name}/{args.truncated}/{args.min_len}_{args.model_size}_zlib_dict.pkl", "rb"))
    # refer_dict = pickle.load(open(f"{args.dir}/{dataset_name}/{args.truncated}/{args.min_len}_{args.model_size}_refer_dict.pkl", "rb"))
    # idx_list = pickle.load(
    #     open(f"{args.dir}/{dataset_name}/{args.truncated}/{args.min_len}_{args.model_size}_idx_list.pkl", "rb"))
    # residual_dict = {}
    # residual_dict[dataset_name] = {"member": [], "nonmember": []}
    # for split in split_set:
    #     residual_dict[dataset_name][split] = [loss_dict[dataset_name][split][i] - refer_dict[dataset_name][split][i]
    #                                           for i in range(len(loss_dict[dataset_name][split]))]
    # grad_dict = pickle.load(open(f"{args.dir}/{dataset_name}/{args.truncated}/{args.min_len}_{args.model_size}_grad_dict.pkl", "rb"))
    # ccd_dict = pickle.load(open(f"{args.dir}/{dataset_name}/{args.truncated}/{args.min_len}_{args.model_size}_ccd_dict.pkl", "rb"))
    # samia_dict = pickle.load(open(f"{args.dir}/{dataset_name}/{args.truncated}/{args.min_len}_{args.model_size}_samia_dict.pkl", "rb"))
    # all_dict = [loss_dict, prob_dict, ppl_dict, mink_plus_dict, zlib_dict, residual_dict, grad_dict, ccd_dict, samia_dict]
    # method_list = ["loss", "prob", "ppl", "mink_plus", "zlib", "refer", "grad", "ccd", "samia"]
    os.makedirs(f"{args.save_dir}_figures/{args.model_size}_{args.min_len}", exist_ok=True)
    for idx, method_name in enumerate(method_list):
        value_dict = pickle.load(open(f"{args.save_dir}/{dataset_name}/{args.truncated}/{args.min_len}_{args.model_size}_{method_name}_dict.pkl", "rb"))
        if method_name == "refer":
            residual_dict = {}
            residual_dict[dataset_name] = {"member": [], "nonmember": []}
            loss_dict = pickle.load(open(f"{args.save_dir}/{dataset_name}/{args.truncated}/{args.min_len}_{args.model_size}_loss_dict.pkl", "rb"))
            refer_dict = pickle.load(open(f"{args.save_dir}/{dataset_name}/{args.truncated}/{args.min_len}_{args.model_size}_refer_dict.pkl", "rb"))
            for split in split_set:
                residual_dict[dataset_name][split] = [
                    loss_dict[dataset_name][split][i] - refer_dict[dataset_name][split][i]
                    for i in range(len(loss_dict[dataset_name][split]))]
            value_dict = residual_dict
        figure_draw(value_dict, method_name, dataset_name, args)
        #mix_distribution(loss_dict, dataset_name, "Loss", args)
        print(f"{method_name} Distribution Similarity Matrix")
        print(idx)
        calculate_mean_var(value_dict, dataset_name, split_set=split_set)
        js_matrix = js_divergence(value_dict, dataset_name, split_set=split_set)
        print(js_matrix)
        ks_matrix, ks_p_value_matrix = ks_hypothesis(value_dict, dataset_name, split_set=split_set)
        print(ks_matrix)
        print(ks_p_value_matrix)
        ws_matrix = wasserstein_distance_caculate(value_dict, dataset_name, split_set=split_set)
        print(ws_matrix)
        df = df._append({'Dataset Name': dataset_name,
                         "Method": method_list[idx],
                 'js_matrix': js_matrix[0][1],
                 'ks_matrix': ks_matrix[0][1],
                 'ks_p_value_matrix': ks_p_value_matrix[0][1],
                 "ws_matrix": ws_matrix[0][1] },
                ignore_index=True)
    return df

def calculate_mean_var(dict, dataset_name, split_set=["train", "valid", "test"]):
    for idx1, set1 in enumerate(split_set):
        values = np.array(dict[dataset_name][set1])
        values = values[np.isnan(values)==False]
        mean = np.mean(values)
        var = np.var(values)
        std = np.std(values)
        kur = kurtosis(values)
        print("The mean, variance, std and kurtosis of {} in {} set are {},  {}, {} and {}".format(dataset_name, set1, mean, var, std, kur))
    return mean, var

def calculate_mink_and_mink_plus(batch_logits, batched_tokenized_inputs):
    batch_input_ids = batched_tokenized_inputs["input_ids"][:, 1:].unsqueeze(-1)
    target_labels = batched_tokenized_inputs["input_ids"].clone()
    target_labels[batched_tokenized_inputs["attention_mask"] == 0] = -100
    batch_probs = F.softmax(batch_logits[:, :-1].float(), dim=-1)
    batch_log_probs = F.log_softmax(batch_logits[:, :-1].float(), dim=-1)
    mask = target_labels[:, 1:] != -100
    mask = mask.unsqueeze(-1)
    batch_token_log_probs = batch_log_probs.gather(dim=-1, index=batch_input_ids).squeeze(-1)
    batch_probs_masked = batch_probs.where(mask, 0)
    batch_log_probs_masked = batch_log_probs.where(mask, 0)
    batch_mu = (batch_probs_masked.float() * batch_log_probs_masked.float()).float().sum(-1)
    batch_sigma =  ((batch_probs_masked.float() * torch.square(torch.where(batch_probs_masked > 0,batch_log_probs_masked.float(),  torch.tensor(0.0, device=batch_log_probs_masked.device, dtype=torch.float32)))).sum(dim=-1)- torch.square(batch_mu.float()).squeeze())
    mask = mask.squeeze(-1)
    batch_mink_plus = (batch_token_log_probs - batch_mu).float() * mask / batch_sigma.float().sqrt()
    token_length = mask.sum(dim=1)
    batch_mink_plus[mask == False] = torch.inf
    batch_token_log_probs[mask == False] = torch.inf
    sorted_mink_plus, _ = torch.sort(batch_mink_plus)
    sorted_mink, _ = torch.sort(batch_token_log_probs)
    batch_mink_plus_avg = []
    batch_mink_avg = []
    for i, length in enumerate(token_length):
        front_values = sorted_mink_plus[i, :length]
        avg = torch.mean(front_values.float()).item()
        batch_mink_plus_avg.append(avg)
        if torch.tensor(avg) == torch.inf:
            pdb.set_trace()
        front_values = sorted_mink[i, :length]
        avg = torch.mean(front_values.float()).item()
        batch_mink_avg.append(avg)
    return batch_mink_plus_avg, batch_mink_avg


def caculate_instance_loss_perplexity_zlib(batch_logits, target_labels, batched_text, model, tokenized_inputs, tokenizer):
    shift_logits = batch_logits[:, :-1, :].contiguous()
    labels = target_labels[:, 1:].contiguous()
    loss_fct = CrossEntropyLoss(reduction='none')
    loss_value_list = []
    ppl_value_list = []
    zlib_value_list = []
    grad_value_list = []
    scaler = GradScaler()
    lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
    instance_losses = lm_loss.view(-1, shift_logits.size(1))
    for idx, i in enumerate(instance_losses):
        torch.cuda.empty_cache()
        loss = i.sum() / sum(i != 0)
        loss.backward(retain_graph=True)
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.detach().norm(2))
        #pdb.set_trace()
        grad_norm = torch.stack(grad_norms).mean()
        model.zero_grad()
        loss_value_list.append(loss.item())
        ppl = torch.exp(loss.float()).item()
        ppl_value_list.append(ppl)
        #pdb.set_trace()
        zlib_value = loss.float().cpu() / (len(zlib.compress(bytes(tokenizer.decode(tokenized_inputs["input_ids"][idx], skip_special_tokens=True), "utf-8"))))
        zlib_value_list.append(zlib_value.item())
        grad_value_list.append(grad_norm.item())
    return loss_value_list, ppl_value_list, zlib_value_list, grad_value_list

def remove_outliers(data, m=2):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    # 找到大于均值 + m * std 和小于均值 - m * std 的离群值
    outliers_high = data > mean + m * std
    outliers_low = data < mean - m * std
    outliers = outliers_high | outliers_low
    # 计算没有离群值的平均值
    mean_without_outliers = np.mean(data[~outliers])
    # 用没有离群值的平均值替换离群值
    data[outliers] = mean_without_outliers
    return data.tolist()


def save_jsonl(data, fpath):
    with open(fpath, 'w+', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
def load_json(fpath):
    with open(fpath) as f:
        return json.load(f)

def clean_text(text):
    words_list = []
    text = text.lower().replace("sentence 2:", '').replace(".", '')
    for sentence in text.split(','):
        words = sentence.replace('["', '')\
                .replace('\\n', '')\
                .replace('"]', '')\
                .replace("['", '')\
                .replace("]", '').split(' ')
        words_list += words
    if '' in words_list: 
        for _ in range(words_list.count('')):
            words_list.remove('')
    return words_list

def process_data_to_json(dataset_name, data):
    if dataset_name == "databricks-dolly-15k-ja":
        data['input'] = (
                '### 指示：\n' + data['instruction'] + '\n\n' +
                '### 文脈 :\n' + data['context'].replace('', '（空）')
        )
        data['input'] = data['input'].str.replace('### 文脈 :\n（空）', '')
        data['output'] = '### 応答:\n' + data['response']
        result = data[['input', 'output']]
        result = result.to_dict(orient='records')
        return result


def load_data(dataset_name, split_name, num_samples):
    if dataset_name in ["alt-e-to-j", "alt-j-to-e","chabsa", "jamp", "janli",
                          "jcommonsenseqa", "jemhopqa", "jmmlu", "jnli", "jsem",
                          "jsick", "jsquad","jsts", "mawps", "niilc"]:
        loaded_data = load_json(f"datasets_contamination/1.3.0/evaluation/{split_name}/{dataset_name}.json")
        random_samples = create_random_samples(loaded_data["samples"], num_samples=num_samples)
    elif dataset_name == "databricks-dolly-15k-ja":
        df = pd.read_json("hf://datasets/llm-jp/databricks-dolly-15k-ja/databricks-dolly-15k-ja.jsonl", lines=True)
        loaded_data = process_data_to_json("databricks-dolly-15k-ja", df)
        random_samples = create_random_samples(loaded_data, num_samples=num_samples)
    elif dataset_name == "oasst1-21k-ja":
        df = pd.read_json("hf://datasets/llm-jp/oasst1-21k-ja/oasst1-21k-ja.jsonl", lines=True)
    elif dataset_name == "databricks-dolly-15k-ja":
        df = pd.read_json("hf://datasets/llm-jp/oasst1-21k-en/oasst1-21k-en.jsonl", lines=True)
        random_samples = df.sample(num_samples)
    return random_samples

def obtain_instruction_time_travel(dataset_name, split_name, model_name=None):
    if model_name == "llm-jp-v2":
        chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '\\n\\n### 指示:\\n' + message['content'] }}{% elif message['role'] == 'system' %}{{ '\\n\\n### 指示:\\n' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\\n\\n### 応答:\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\\n\\n### 応答:\\n' }}{% endif %}{% endfor %}"
    else:
        chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{message['content'] }}{% elif message['role'] == 'system' %}{{message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\\n### 回答:\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\\n### 回答:\\n' }}{% endif %}{% endfor %}"
    if dataset_name in ["jnli", "jsick", "jamp","janli"]:
        guided_chat = [
            {"role": "system",
             "content": f"次の文1は、{dataset_name}データセットの{split_name}分割から提供されています。\n文1の後にあるラベルは、文1と文2の間の論理的な関係を示します。\n文2はデータセット内のインスタンスと完全に一致する必要があります。\nデータセットに表示された通りに、文2を完成させてください。必ず文2のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "以下のラベルが文1と文2の論理的関係を示すように、文1を基に文2を完成させる。\n必ず文2のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["alt-e-to-j", "alt-j-to-e"]:
        if dataset_name == "alt-e-to-j":
            info = "英語から日本語"
        else:
            info = "日本語から英語"
        guided_chat = [
            {"role": "system",
             "content": f"次の文は、{dataset_name}データセットの{split_name}分割から提供されています。\n原文の前半と原文全体の{info}への翻訳文は提供されています。\nデータセットに表示された通りに、原文の後半を完成させてください。必ずその原文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": f"原文の前半と原文全体の{info}への翻訳文は提供されています。必ずその原文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jemhopqa"]:
        guided_chat = [
                    {"role": "system",
                     "content": f"次の質問文の前半と答えは、{dataset_name}データセットの{split_name}分割から提供されています。\n質問文の下の半分を出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、質問文の下の半分を出力してください。それ以外には何も含めないことを厳守してください。"},
                    {"role": "user", "content": ""},
                ]
        general_chat = [
            {"role": "system",
             "content": f"質問文の半分と答えは提供されいます。\n質問文の下の半分を出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、質問文の下の半分を出力してください。それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jmmlu"]:
        guided_chat = [
                    {"role": "system",
                     "content": f"次の質問文と答えは、{dataset_name}データセットの{split_name}分割から提供されています。\nその質問文と答えを参考し、ABCDという四つの選択肢を出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、選択肢を出力してください。必ずABCDの選択肢のみを出力し、それ以外には何も含めないことを厳守してください。"},
                    {"role": "user", "content": ""},
                ]
        general_chat = [
            {"role": "system",
             "content": f"質問文と答えは提供されています。\nその質問文と答えを参考し、ABCDという四つの選択肢を出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、選択肢を出力してください。必ずABCDの選択肢のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jcommonsenseqa"]:
        guided_chat = [
            {"role": "system",
             "content": f"次の質問文と答えは、{dataset_name}データセットの{split_name}分割から提供されています。\nその質問文と答えを参考し、この答えと質問文に合う01234という五つの選択肢を順番に出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、選択肢を出力してください。必ず01234の選択肢のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": f"質問文と答えは提供されています。\nその質問文と答えを参考し、この答えと質問文に合う01234という五つの選択肢を順番に出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、選択肢を出力してください。必ず01234の選択肢のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jsts"]:
        guided_chat = [
            {"role": "system",
             "content": f"次の文1は、{dataset_name}データセットの{split_name}分割から提供されています。\n文1の後にある数字は、文1と文2の間の類似度を示します。0.0に近いほど文ペアの意味が異なり、5.0に近いほど文ペアの意味が似ていることを表しています。\n文2はデータセット内のインスタンスと完全に一致する必要があります。\nデータセットに表示された通りに、文2を完成させてください。文1とその類似度を使って、必ず文2のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "文1の後にある数字は、文1と文2の間の類似度を示します。0.0に近いほど文ペアの意味が異なり、5.0に近いほど文ペアの意味が似ていることを表しています。\n文1とその類似度を使って、必ず文2のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["niilc"]:
        guided_chat = [
            {"role": "system",
             "content": f"次の質問に対する答えと質問文の前半は、{dataset_name}データセットの{split_name}分割から提供されています。\nデータセットに表示された通りに、質問文の後半を完成させてください。\nその文はデータセット内のインスタンスと完全に一致する必要があります。必ず質問文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "質問に対する答えと質問文の前半は提供されています。\n必ず質問文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["mawps"]:
        guided_chat = [
            {"role": "system",
             "content": f"次の計算問題に対する答えと質問文の前半は、{dataset_name}データセットの{split_name}分割から提供されています。\データセットに表示された通りに、質問文の後半を完成させてください。必ずその質問文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "計算問題に対する答えと質問文の前半は提供されています。\n必ずその質問文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jsquad"]:
        guided_chat = [
            {"role": "system",
             "content": f"文章と文章に対する答えは、{dataset_name}データセットの{split_name}分割から提供されています。\nデータセットに表示された通りに、その文章と答えに合う質問文を書いてください。必ず質問文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "文章と文章に対する答えは提供されています。\nその文章と答えに合う質問文を書いてください。必ず質問文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jsem"]:
        guided_chat = [
            {"role": "system",
             "content": f"前提と、前提と仮説の関係の答えは、{dataset_name}データセットの{split_name}分割から提供されています。\nその答えはyes、no、unknown、undefの中からの答えと提供されています\nデータセットに表示された通りに、仮説文を完成させてください。必ず仮説文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "前提と、前提と仮説の関係の答えはyes、no、unknown、undefから提供されています\nデータセットに表示された通りに、仮説文を書いてください。必ず仮説文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["chabsa"]:
        guided_chat = [
            {"role": "system",
             "content": f"与えられた文章の前半と、全体の文章から抽出された固有表現で書かれたターゲットの名前と、それぞれの名前に対するpositive、neutral、negativeの極性は、{dataset_name}データセットの{split_name}分割から提供されています。\nデータセットに表示された通りに、未完成の文章の後半を書いてください。必ず文章の後半の文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "与えられた文章の前半と、全体の文章から抽出された固有表現で書かれたターゲットの名前と、それぞれの名前に対するpositive、neutral、negativeの極性は提供されています\n未完成の文章の後半を書いてください。必ず文章の後半の文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name == "databricks-dolly-15k-ja":
        if model_name == "llm-jp-v2":
            chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '' + message['content'] }}{% elif message['role'] == 'system' %}{{ '' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '' }}{% endif %}{% endfor %}"
        else:
            chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{message['content'] }}{% elif message['role'] == 'system' %}{{message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '' }}{% endif %}{% endfor %}"
        guided_chat = [
            {"role": "system",
             "content": "最初の指示文と文脈文と応答の一部が提供されます。それらの情報に基づいて、データセット中の応答文の原形の残った文を完成させなさい。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat,general_chat, chat_template
def obtain_instruction_naive(dataset_name, split_name, model_name=None):
    if model_name == "llm-jp-v2":
        chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '' + message['content'] }}{% elif message['role'] == 'system' %}{{ '' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '' }}{% endif %}{% endfor %}"
    else:
        chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{message['content'] }}{% elif message['role'] == 'system' %}{{message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '' }}{% endif %}{% endfor %}"
    if dataset_name == "databricks-dolly-15k-ja":
        general_chat = [
            {"role": "system",
             "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"},
            {"role": "user", "content": ""},
        ]
    return general_chat, chat_template


def formalize_input_time_travel(dataset_name,guided_chat, general_chat, inst_type, example):
    if dataset_name in ["jnli", "jsick", "jamp", "janli"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        procesesd_sent1 = example['input'].split('\n')[0].replace('前提：', '')
        sent1 = f"文1: {procesesd_sent1}"
        sent2 = "文2: "+example['input'].split("\n")[1].replace("仮説：", "")
        label = "含意" if example['output'] == "entailment" else "矛盾" if example['output'] == "contradiction" else "中立"
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\nラベル:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\nラベル:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["alt-e-to-j", "alt-j-to-e"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        if dataset_name == "alt-e-to-j":
            sent1 = " ".join(example['input'].split(" ")[:len(example['input'].split(" ")) // 2])
            sent1 = sent1.strip()
            label = example['output']
            sent2 = " ".join(example['input'].split(" ")[len(example['input'].split(" "))//2:])
            sent2= sent2.strip()
        else:
            sent1 = example['input'][:len(example['input']) // 2]
            sent1 = sent1.strip()
            label = example['output']
            sent2 = example['input'][len(example['input']) // 2:]
            sent2 = sent2.strip()
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"原文:{sent1}\n翻訳文:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"原文:{sent1}\n翻訳文:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["jemhopqa", "niilc", "mawps"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1 = example['input'][:len(example['input'])//2]
        label = example['output']
        sent2 = example['input'][len(example['input'])//2:]
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\n答え:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\n答え:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["jsts"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1, sent2 = example['input'].split('\n')
        label = example['output']
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1.strip()}\n類似度:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1.strip()}\n類似度:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["jcommonsenseqa", "jmmlu"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1 = example['input'].split("\n")[0]
        label = example['output']
        sent2 = example['input'].split("\n")[1]
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\n答え:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\n答え:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["jsquad"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1, sent2 = example['input'].split('\n')
        label = example['output']
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\n答え:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\n答え:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["jsem"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1, sent2 = example['input'].split('\n')
        label = example['output']
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\n前提と仮説の関係の答え:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\n前提と仮説の関係の答え:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["chabsa"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1, sent2 = example['input'][:len(example['input'])//2], example['input'][len(example['input'])//2:]
        label = example['output']
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\nターゲットの名前とそれぞれの極性:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\nターゲットの名前とそれぞれの極性:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name == "databricks-dolly-15k-ja":
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1 = example['input']
        prev_half = example["output"][:len(example["output"])//2]
        sent2 = example["output"][len(example["output"])//2:]
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\n"+prev_half
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\n"+prev_half
        return chat, f"{sent1}\n"+prev_half, sent2, instruction

def formalize_input_baseline(dataset_name, general_chat, example):
    if dataset_name == "databricks-dolly-15k-ja":
        sent1 = example['input']
        prev_half = example["output"][:len(example["output"])//2]
        sent2 = example["output"][len(example["output"])//2:]
        chat = general_chat
        chat[1]["content"] = f"{sent1}\n"+prev_half
        return chat, f"{sent1}\n"+prev_half, sent2, chat[0]["content"]+f"{sent1}\n"+prev_half
    

def calculate_perplexity(output, tokenized_input):
    # 提取生成的logits和生成的序列
    logits = torch.stack(output.scores, dim=1)[0]  # (sequence_length, vocab_size)
    generated_seq = output.sequences[0]  # (sequence_length_with_prompt + generated_tokens)
    # 提取生成部分的目标序列
    input_length = tokenized_input.size(-1)
    target_seq = generated_seq[input_length:]
    # 确保logits和目标序列对齐
    assert logits.size(0) == target_seq.size(0), "Logits and target sequence length must match."
    # 计算交叉熵损失和困惑度
    perplexities = []
    with torch.no_grad():
        for i in range(logits.size(0)):  # iterate over time steps
            # 获取当前 step 的 logits 和 label
            current_logits = logits[i, :].unsqueeze(0)  # (1, vocab_size)
            current_label = target_seq[i].unsqueeze(0)  # (1,)
            # 计算交叉熵损失
            loss = F.cross_entropy(current_logits, current_label, reduction='none')
            # 计算困惑度
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
    return perplexities

def calculate_memorization_score(output, tokenized_input, continuation, tokenizer):
    # 提取生成的序列
    generated_seq = output.sequences[0]  # (sequence_length_with_prompt + generated_tokens)
    # 提取生成部分的目标序列
    input_length = tokenized_input.size(-1)
    target_seq = generated_seq[input_length:]
    # 计算生成部分的准确率
    correct = 0
    for i in range(min(len(continuation), target_seq.size(0))):
        if target_seq[i] == tokenizer.eos_token_id:
            break  # 如果遇到eos_token，停止计数
        if target_seq[i] == continuation[i]:
            correct += 1
    return correct / target_seq.size(0) if target_seq.size(0) > 0 else 0
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def do_plot(prediction, answers, sweep_fn=sweep, metric='auc', legend="", output_dir=None):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """
    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr<.05)[0][-1]]
    # bp()
    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@5%%FPR of %.4f\n'%(legend, auc,acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc

    plt.plot(fpr, tpr, label=legend+metric_text)
    return legend, auc,acc, low

def fig_fpr_tpr(all_output, output_dir):
    print("output_dir", output_dir)
    answers = []
    metric2predictions = defaultdict(list)
    for ex in all_output:
        answers.append(ex["label"])
        for metric in ex["pred"].keys():
            if ("raw" in metric) and ("clf" not in metric):
                continue
            metric2predictions[metric].append(ex["pred"][metric])

    plt.figure(figsize=(4, 3))
    with open(f"{output_dir}/auc.txt", "w") as f:
        for metric, predictions in metric2predictions.items():
            legend, auc, acc, low = do_plot(predictions, answers, legend=metric, metric='auc', output_dir=output_dir)
            f.write('%s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f\n' % (legend, auc, acc, low))

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig(f"{output_dir}/auc.png")

def get_dataset_list(args):
    if args.dataset_name == "WikiMIA":
        return ["WikiMIA64", "WikiMIA128", "WikiMIA256", "WikiMIAall"]
    elif args.dataset_name == "temporalarxiv":
        return ["temporalarxiv_2020_08", "temporalarxiv_2021_01", "temporalarxiv_2021_06",
                "temporalarxiv_2022_01", "temporalarxiv_2022_06", "temporalarxiv_2023_01", "temporalarxiv_2023_06"]
    elif args.dataset_name=="online_all":
        return ["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                    "pubmed_central", "wikipedia_(en)", "full_pile"]
    elif args.dataset_name=="local_all":
        if args.truncated == "truncated":
            return['Wikipedia (en)', "USPTO Backgrounds", "StackExchange", 'PubMed Central', "Pile-CC", "HackerNews",
                   "Github", "FreeLaw", "EuroParl",'DM Mathematics',"ArXiv",]
        elif args.truncated == "untruncated":
            return ['Wikipedia (en)', "USPTO Backgrounds", "StackExchange", "Pile-CC", "Github", "FreeLaw"]
        elif "relative" in args.dir:
            return ["ArXiv", "Enron Emails", "FreeLaw", 'Gutenberg (PG-19)', 'NIH ExPorter', "Pile-CC", 'PubMed Central',
            'Ubuntu IRC', 'Wikipedia (en)', 'DM Mathematics', "EuroParl", "Github", "HackerNews", "PhilPapers",
            "PubMed Abstracts", "StackExchange"]
    else:
        return [args.dataset_name]

def obtain_dataset(dataset_name, args):
    if args.local_data == True and "temporalarxiv" not in dataset_name and "WikiMIA" not in dataset_name:
        dataset = form_dataset(dataset_name, args)
    elif "WikiMIA" in dataset_name:
        dataset = form_dataset(dataset_name, args)
    elif "temporalarxiv" in dataset_name:
        dataset = load_dataset("iamgroot42/mimir", "temporal_arxiv",
                               split=dataset_name.replace("temporalarxiv_", ""))
    else:
        dataset = load_dataset("iamgroot42/mimir", dataset_name,
                               split="ngram_13_0.2") if dataset_name != "full_pile" else load_dataset(
            "iamgroot42/mimir",
            "full_pile",
            split="none")
    return dataset

def load_model_and_tokenizer(args):
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{args.model_size}-deduped",
        revision="step143000",
        cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
        torch_dtype=torch.bfloat16,
    ).cuda(args.cuda).eval()
    model = model.to_bettertransformer()

    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{args.model_size}-deduped",
        revision="step143000",
        cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
    )
    return model, tokenizer
