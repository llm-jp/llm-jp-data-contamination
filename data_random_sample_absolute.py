import os
import pdb
from tqdm import tqdm
import torch
import random
import numpy as np
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset
import argparse

random.seed(42)

# 初始化 parser
parser = argparse.ArgumentParser()
parser.add_argument("--list", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--sample_size", type=int, default=1000)
parser.add_argument("--select_method", type=str, default="nontruncate", choices=["truncate", "nontruncate"])
parser.add_argument("--relative_length", type=bool)
args = parser.parse_args()


def load_test_data(test_folder, test_files, min_length, max_length, sample_size, filter_test, tokenizer, batch_size):
    if len(test_files) > 1:
        sampled_test_files = random.sample(test_files, num_samples)
        if filter_test:
            test_data = load_and_filter_data(sampled_test_files, test_folder, min_length, max_length, sample_size,
                                             tokenizer, batch_size)
        else:
            merged_test_data = []
            for file in sampled_test_files:
                dataset = torch.load(os.path.join(test_folder, file))
                merged_test_data.extend(dataset)
            if len(merged_test_data) > sample_size:
                test_data = random.sample(merged_test_data, sample_size)
            else:
                test_data = merged_test_data
    else:
        test_dataset = torch.load(os.path.join(test_folder, f"test_{dataset_name}_0.pt"))
        test_data = filter_data(test_dataset, min_length, max_length, tokenizer, batch_size)
        if len(test_data) > sample_size:
            test_data = random.sample(test_data, sample_size)
    return test_data, test_dataset


def filter_data(data, min_length, max_length, tokenizer, args):
    """批量过滤文本长度在给定Token数量范围的数据"""
    filtered_data = []
    for i in tqdm(range(0, len(data), args.batch_size)):
        batch = data[i:i + args.batch_size]
        texts = [item for item in batch]
        # if args.tokenize_method == "tokenizer":
        #     tokenized_batch = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=2048).to("cuda:1")
        #     lengths = tokenized_batch['attention_mask'].cuda(1).sum(dim=1)
        # elif args.tokenize_method == "space":
        lengths = [len(text.split()) for text in texts]
        if args.select_method == "nontruncate":
            valid_indices = (np.array(lengths) >= min_length) & (np.array(lengths) <= max_length)
            filtered_data.extend([batch[j] for j in range(len(batch)) if valid_indices[j]])
        elif args.select_method == "truncate":
            valid_indices = (np.array(lengths) >= max_length)
            filtered_data.extend([" ".join(batch[j].split()[:max_length]) for j in range(len(batch)) if valid_indices[j]])
    return filtered_data


def load_and_filter_data(dataset, min_length, max_length, tokenizer, args):
    """filtering and load"""
    merged_data = []
    filtered_data = filter_data(dataset, min_length, max_length, tokenizer, args)
    merged_data.extend(filtered_data)
    if len(merged_data) > args.sample_size:
        return random.sample(merged_data, args.sample_size)
    return merged_data


def compute_length_percentiles(data, tokenizer, batch_size):
    """计算数据长度的百分位数"""
    lengths = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        texts = [item for item in batch]
        #tokenized_batch = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=2048).to(
        #    "cuda:1")
        #batch_lengths = tokenized_batch['attention_mask'].cuda(1).sum(dim=1).cpu().numpy()
        batch_lengths = [len(text.split()) for text in texts]
        lengths.extend(batch_lengths)
    percentiles = np.percentile(lengths, np.arange(0, 110, 10))
    return percentiles


# 创建数据集名称列表
if args.list == 1:
    datalist = ['Gutenberg (PG-19)', 'NIH ExPorter', "Pile-CC", "ArXiv", "Enron Emails", "FreeLaw"]
elif args.list == 2:
    datalist = ['PubMed Central', 'Ubuntu IRC', 'Wikipedia (en)', 'DM Mathematics', "EuroParl", "Github"]
else:
    datalist = ["HackerNews", "PhilPapers",	 "PubMed Abstracts",   "StackExchange",    "USPTO Backgrounds"]

# 处理每个数据集
for dataset_name in datalist:
    train_folder = "/model/pile/by_dataset/"
    test_folder = "/model/pile/by_dataset/"

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b-deduped", revision="step143000",
                                              cache_dir="./pythia-12b-deduped/step143000")
    tokenizer.pad_token = tokenizer.eos_token

    # Step 1: see how many data files exist
    train_files = [f for f in os.listdir(train_folder) if f.startswith(f"train_{dataset_name}_")]
    test_files = [f for f in os.listdir(test_folder) if f.startswith(f"test_{dataset_name}_")]

    # Step 2: sample train_{dataset_name}_x
    num_samples = 5 if len(train_files) > 5 else len(train_files)
    sampled_train_files = random.sample(train_files, num_samples)

    # Step 3 & 4: merge data and take 20000 samples
    test_dataset_full = []
    for file in test_files:
        dataset = torch.load(os.path.join(test_folder, file))
        test_dataset_full.extend(dataset)
    train_dataset_full = []
    for file in sampled_train_files:
        dataset = torch.load(os.path.join(train_folder, file))
        train_dataset_full.extend(dataset)
    percentiles = compute_length_percentiles(test_dataset_full, tokenizer, args.batch_size)
    full_nonmember_data = test_dataset_full
    if args.relative_length:
        length_list = percentiles.tolist()
        enumerate_length =  len(length_list) - 1
    else:
        length_list = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, "rest"]
        enumerate_length = len(length_list)
    pdb.set_trace()
    for i in range(enumerate_length):
        member_data = []
        nonmember_data = []
        if args.relative_length:
            min_length = length_list[i]
            max_length = length_list[i + 1]
        else:
            if length_list[i] == 0:
                min_length = 5
                max_length = length_list[i + 1]
            elif length_list[i+1] == "rest":
                min_length = 1000
                max_length = 100000000000
            else:
                min_length = length_list[i]
                max_length = min_length + 100
        filtered_member_data = load_and_filter_data(train_dataset_full, train_folder, min_length, max_length, tokenizer, args)
        filtered_nonmember_data = load_and_filter_data(test_dataset_full, test_folder, min_length, max_length, tokenizer, args)

        member_data.extend(filtered_member_data)
        nonmember_data.extend(filtered_nonmember_data)

        # 创建数据集
        train_dataset = Dataset.from_dict({"data": member_data})
        test_dataset_short = Dataset.from_dict({"data": nonmember_data})
        full_test_dataset = Dataset.from_dict({"data": full_nonmember_data})

        # 创建 DatasetDict 对象
        dataset = DatasetDict({
            'member': train_dataset,
            'nonmember': test_dataset_short,
            "full_nonmember": full_test_dataset
        })

        # 保存数据集
        if args.relative_length:
            os.makedirs(f"./relative_filtered_dataset/{i}/{dataset_name}", exist_ok=True)
            dataset.save_to_disk(f"./relative_filtered_dataset/{i}/{dataset_name}")
        else:
            os.makedirs(f"./absolute_filtered_dataset/{min_length}_{max_length}/{dataset_name}", exist_ok=True)
            dataset.save_to_disk(f"./absolute_filtered_dataset/{min_length}_{max_length}/{dataset_name}")
        print(dataset)