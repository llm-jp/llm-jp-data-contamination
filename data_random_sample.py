import os
import pdb
from tqdm import tqdm
import torch
import random
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset


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
        if len(test_data) < 100:
            test_data = test_dataset
        if len(test_data) > sample_size:
            test_data = random.sample(test_data, sample_size)
    return test_data


def filter_data(data, min_length, max_length, tokenizer, batch_size):
    """批量过滤文本长度在给定Token数量范围的数据"""
    filtered_data = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        texts = [item for item in batch]
        tokenized_batch = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=2048).to(
            "cuda:1")
        # pdb.set_trace()
        # use attention_mask to obtain the length of each text
        lengths = tokenized_batch['attention_mask'].cuda(1).sum(dim=1)
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
#"ArXiv",
for dataset_name in ["Wikipedia (en)", "PubMed", "USPTO", "FreeLaw", "PubMed Central"
                     "Enron Emails", "HackerNews", "NIH", "DM Mathematics", "Ubuntu IRC", "EuroParil", "PhilPapers", "Gutenberg(PG-19)"]:
    train_folder = "/model/pile/by_dataset/"
    test_folder = "/model/pile/by_dataset/"
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-12b-deduped",
        revision="step143000",
        cache_dir=f"./pythia-12b-deduped/step143000",
    )
    tokenizer.pad_token = tokenizer.eos_token
    for min_length in [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]:
        max_length = min_length + 100  # token
        batch_size = 100 #deocder batch size

        # Step 1: see how many data files exist
        train_files = [f for f in os.listdir(train_folder) if f.startswith(f"train_{dataset_name}_")]
        test_files = [f for f in os.listdir(test_folder) if f.startswith(f"test_{dataset_name}_")]

        # Step 2: sample train_{dataset_name}_x
        num_samples = 5 if len(train_files) > 5 else len(train_files)
        sampled_train_files = random.sample(train_files, num_samples)

        # Step 3 & 4: merge data and take 20000 samples
        train_data = load_and_filter_data(sampled_train_files, train_folder, min_length, max_length, 20000, tokenizer, batch_size)

        # whether to filter the test data since the test data is rare and may not reach 20000 sample size
        filter_test = True
        test_data = load_test_data(test_folder, test_files, min_length, max_length, 20000, filter_test, tokenizer, batch_size)

        # create data dict
        dataset = DatasetDict({
            'member': Dataset.from_dict({'data': train_data}),
            'nonmember': Dataset.from_dict({'data': test_data}),
        })
        # save the dataset
        os.makedirs(f"./filtered_dataset/{min_length}_{max_length}/{dataset_name}", exist_ok=True)
        dataset.save_to_disk(f"./filtered_dataset/{min_length}_{max_length}/{dataset_name}")
        print(dataset)
