import os
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from concurrent.futures import ThreadPoolExecutor

random.seed(42)


def load_test_data(test_folder, test_files, min_length, max_length, sample_size, filter_test, tokenizer, batch_size, dataset_name):
    test_data, test_dataset = [], []
    if len(test_files) > 1:
        sampled_test_files = random.sample(test_files, len(test_files))
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
        texts = batch
        tokenized_batch = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=2048).to(
            "cuda:1")
        lengths = tokenized_batch['attention_mask'].sum(dim=1)
        valid_indices = (lengths >= min_length) & (lengths <= max_length)
        filtered_data.extend([batch[j] for j in range(len(batch)) if valid_indices[j]])
    return filtered_data


def load_and_filter_data(files, folder, min_length, max_length, sample_size, tokenizer, batch_size):
    """filtering and load"""
    merged_data = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(load_and_filter_file, os.path.join(folder, file), min_length, max_length, tokenizer,
                            batch_size) for file in files]
        for future in futures:
            filtered_data = future.result()
            merged_data.extend(filtered_data)
            if len(merged_data) > sample_size:
                return random.sample(merged_data, sample_size)
    return merged_data


def load_and_filter_file(filepath, min_length, max_length, tokenizer, batch_size):
    dataset = torch.load(filepath)
    return filter_data(dataset, min_length, max_length, tokenizer, batch_size)


# 处理数据集名称和长度限制
dataset_names = ["ArXiv", "Wikipedia (en)", "PubMed Abstracts", "USPTO Backgrounds",
                 "FreeLaw", "PubMed Central", "Enron Emails", "HackerNews",
                 "NIH", "DM Mathematics", "Ubuntu IRC", "EuroParl", "PhilPapers", "Gutenberg (PG-19)"]

min_lengths = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]

# 打开缓存字典进行共享tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-12b-deduped", revision="step143000", cache_dir="./pythia-12b-deduped/step143000"
)
tokenizer.pad_token = tokenizer.eos_token


# 创建并行处理函数
def process_dataset(dataset_name):
    train_folder = "/model/pile/by_dataset/"
    test_folder = "/model/pile/by_dataset/"
    print(dataset_name)
    for min_length in min_lengths:
        max_length = min_length + 100
        batch_size = 100

        # 处理文件列表和采样
        train_files = [f for f in os.listdir(train_folder) if f.startswith(f"train_{dataset_name}_")]
        test_files = [f for f in os.listdir(test_folder) if f.startswith(f"test_{dataset_name}_")]

        num_samples = 5 if len(train_files) > 5 else len(train_files)
        sampled_train_files = random.sample(train_files, num_samples)

        # 过滤和加载数据
        train_data = load_and_filter_data(sampled_train_files, train_folder, min_length, max_length, 20000, tokenizer,
                                          batch_size)
        filter_test = True
        test_data, test_dataset = load_test_data(test_folder, test_files, min_length, max_length, 20000, filter_test,
                                                 tokenizer, batch_size, dataset_name)

        # 构建 Dataset 和 DatasetDict
        train_dataset = Dataset.from_dict({"text": train_data})
        test_dataset_short = Dataset.from_dict({"text": test_data})
        full_test_dataset = Dataset.from_dict({"text": test_dataset})

        dataset = DatasetDict({
            'member': train_dataset,
            'nonmember': test_dataset_short,
            "full_nonmember": full_test_dataset
        })

        # 保存 DatasetDict
        save_path = f"./filtered_dataset/{min_length}_{max_length}/{dataset_name}"
        os.makedirs(save_path, exist_ok=True)
        dataset.save_to_disk(save_path)
        print(dataset)


# 多线程处理所有数据集
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(process_dataset, dataset_name) for dataset_name in dataset_names]
    for future in futures:
        future.result()