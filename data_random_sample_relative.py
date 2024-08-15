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


def filter_data(data, min_length, max_length, tokenizer, batch_size):
    """批量过滤文本长度在给定Token数量范围的数据"""
    filtered_data = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        texts = [item for item in batch]
        # tokenized_batch = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=2048).to(
        #     "cuda:1")
        # lengths = tokenized_batch['attention_mask'].cuda(1).sum(dim=1)
        lengths = [len(text.split()) for text in texts]
        valid_indices = (lengths >= min_length) & (lengths <= max_length)
        filtered_data.extend([batch[j] for j in range(len(batch)) if valid_indices[j]])
    return filtered_data


def load_and_filter_data(files, folder, min_length, max_length, sample_size, tokenizer, batch_size):
    """filtering and load"""
    merged_data = []
    for file in files:
        dataset = torch.load(os.path.join(folder, file))
        filtered_data = filter_data(dataset, min_length, max_length, tokenizer, batch_size)
        merged_data.extend(filtered_data)
    if len(merged_data) > sample_size:
        return random.sample(merged_data, sample_size)
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
    datalist = ["ArXiv" 'Enron Emails', "FreeLaw", 'Gutenberg (PG-19)', 'NIH ExPorter', "Pile-CC"]
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

    percentiles = compute_length_percentiles(test_dataset_full, tokenizer, args.batch_size)

    member_data = []
    nonmember_data = []
    full_nonmember_data = test_dataset_full

    for i in range(len(percentiles) - 1):
        min_length = percentiles[i]
        max_length = percentiles[i + 1]

        filtered_member_data = load_and_filter_data(sampled_train_files, train_folder, min_length, max_length,
                                                    args.sample_size, tokenizer, args.batch_size)
        filtered_nonmember_data = load_and_filter_data(test_files, test_folder, min_length, max_length, args.sample_size,
                                                       tokenizer, args.batch_size)

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
        os.makedirs(f"./relative_filtered_dataset/{i*10}/{dataset_name}", exist_ok=True)
        dataset.save_to_disk(f"./relative_filtered_dataset/{i*10}/{dataset_name}")

        print(dataset)