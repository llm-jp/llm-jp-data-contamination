import os
import pdb
from tqdm import tqdm
import torch
import random
import numpy as np
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset
import argparse



def filter_data(data, min_length, max_length, args):
    """批量过滤文本长度在给定Token数量范围的数据"""
    filtered_data = []
    for i in tqdm(range(0, len(data), args.batch_size)):
        batch = data[i:i + args.batch_size]
        texts = [item for item in batch]
        lengths = [len(text.split()) for text in texts]
        if args.select_method == "nontruncate":
            valid_indices = (np.array(lengths) >= min_length) & (np.array(lengths) <= max_length)
            filtered_data.extend([batch[j] for j in range(len(batch)) if valid_indices[j]])
        elif args.select_method == "truncate" and args.relative_length == "False":
            valid_indices = (np.array(lengths) >= min_length)
            filtered_data.extend([" ".join(batch[j].split()[:max_length]) for j in range(len(batch)) if valid_indices[j]])
    return filtered_data


def load_and_filter_data(dataset, min_length, max_length, args):
    """filtering and load"""
    merged_data = []
    filtered_data = filter_data(dataset, min_length, max_length, args)
    merged_data.extend(filtered_data)
    if len(merged_data) > args.sample_size:
        return random.sample(merged_data, args.sample_size)
    return merged_data


def compute_length_percentiles(data, batch_size):
    """计算数据长度的百分位数"""
    lengths = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        texts = [item for item in batch]
        batch_lengths = [len(text.split()) for text in texts]
        lengths.extend(batch_lengths)
    percentiles = np.percentile(lengths, np.arange(0, 110, 10))
    return percentiles
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

parser = argparse.ArgumentParser()
parser.add_argument("--list", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--sample_size", type=int, default=1000)
parser.add_argument("--select_method", type=str, default="nontruncate", choices=["truncate", "nontruncate"])
parser.add_argument("--relative_length", type=str, default="False")
args = parser.parse_args()
dataset_names = ["ArXiv", "DM Mathematics", "Enron Emails", "EuroParl", "FreeLaw", "Github", "Gutenberg (PG-19)",
                "HackerNews", "NIH ExPorter", "PhilPapers", "PubMed Abstracts", "PubMed Central", "Pile-CC",
                "StackExchange", "Ubuntu IRC", "USPTO Backgrounds", "Wikipedia (en)"]
dataset_num = {"ArXiv":4841, "DM Mathematics": 3929, "Enron Emails": 1957, "EuroParl": 290, "FreeLaw": 10195,
               "Github": 36532, "Gutenberg (PG-19)": 140, "HackerNews": 3251, "NIH ExPorter":3709, "PhilPapers": 132,
               "PubMed Abstracts": 59766, "PubMed Central": 11888, "Pile-CC": 105582, "StackExchange": 60328, "Ubuntu IRC":43,
               "USPTO Backgrounds": 22802, "Wikipedia (en)": 34989}
os.makedirs("pythia-train", exist_ok=True)
seed_list = [10345]
for idx, seed in enumerate(seed_list):
    random.seed(seed)

    # 初始化 parser
    # 处理每个数据集
    for dataset_name in dataset_names:
        train_folder = "/model/pile/by_dataset/"

        # Step 1: see how many data files exist
        train_files = [f for f in os.listdir(train_folder) if f.startswith(f"train_{dataset_name}_")]

        # Step 2: sample train_{dataset_name}_x
        num_samples = 5 if len(train_files) > 5 else len(train_files)
        sampled_train_files = random.sample(train_files, num_samples)
        # Step 3 & 4: merge data and take 20000 samples
        train_dataset_full = []
        for file in sampled_train_files:
            dataset = torch.load(os.path.join(train_folder, file))
            train_dataset_full.extend(dataset)
        random.sample(train_dataset_full, dataset_num[dataset_name])
        # Step 5: save the sampled dataset
        torch.save(train_dataset_full, f"pythia-train/train_{dataset_name}_0.pt")
        #pdb.set_trace()

