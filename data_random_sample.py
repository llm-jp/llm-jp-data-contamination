import os
import pdb
from tqdm import tqdm
import torch
import random
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset

dataset_name = "ArXiv"
train_folder = "/model/pile/by_dataset/"
test_folder = "/model/pile/by_dataset/"
min_length = 50  # 最小长度（Token数量）
max_length = 5000  # 最大长度（Token数量）
batch_size = 100
tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-12b-deduped",
        revision="step143000",
        cache_dir=f"./pythia-12b-deduped/step143000",
    )
tokenizer.pad_token = tokenizer.eos_token
def filter_data(data, min_length, max_length, tokenizer, batch_size):
    """批量过滤文本长度在给定Token数量范围的数据"""
    filtered_data = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        texts = [item for item in batch]
        tokenized_batch = tokenizer(texts, truncation=True, padding='longest', return_tensors="pt")
        # 使用 attention_mask 获得有效 Token 的长度
        lengths = tokenized_batch['attention_mask'].sum(dim=1)
        valid_indices = (lengths >= min_length) & (lengths <= max_length)
        filtered_data.extend([batch[j] for j in range(len(batch)) if valid_indices[j]])
    return filtered_data

def load_and_filter_data(files, folder, min_length, max_length, sample_size, tokenizer, batch_size):
    """加载并过滤数据，然后随机抽样指定数量的样本"""
    merged_data = []
    for file in files:
        dataset = torch.load(os.path.join(folder, file))
        filtered_data = filter_data(dataset, min_length, max_length, tokenizer, batch_size)
        merged_data.extend(filtered_data)
    if len(merged_data) > sample_size:
        return random.sample(merged_data, sample_size)
    return merged_data

# Step 1: 检测文件夹下有多少train_{dataset_name}_x
train_files = [f for f in os.listdir(train_folder) if f.startswith(f"train_{dataset_name}_")]
test_files = [f for f in os.listdir(test_folder) if f.startswith(f"test_{dataset_name}_")]

# Step 2: 随机抽样多个train_{dataset_name}_x
num_samples = 3  # 假设我们抽样3个文件，这个数字可以根据需要调整
sampled_train_files = random.sample(train_files, num_samples)

# Step 3 & 4: 把多个train_{dataset_name}_x合并并在合并train_{dataset_name}_x中，随机采样20000个样本
train_data = load_and_filter_data(sampled_train_files, train_folder, min_length, max_length, 20000, tokenizer, batch_size)

# Step 5: 处理test数据

def load_test_data(test_folder, test_files, min_length, max_length, sample_size, filter_test, tokenizer, batch_size):
    if len(test_files) > 1:
        sampled_test_files = random.sample(test_files, num_samples)
        if filter_test:
            test_data = load_and_filter_data(sampled_test_files, test_folder, min_length, max_length, sample_size, tokenizer, batch_size)
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
        if filter_test:
            test_data = filter_data(test_dataset, min_length, max_length, tokenizer, batch_size)
            if len(test_data) > sample_size:
                test_data = random.sample(test_data, sample_size)
        else:
            if len(test_dataset) > sample_size:
                test_data = random.sample(test_dataset, sample_size)
            else:
                test_data = test_dataset
    return test_data

# 控制参数，用以选择是否过滤test数据
filter_test = True
test_data = load_test_data(test_folder, test_files, min_length, max_length, 20000, filter_test, tokenizer, batch_size)

# 创建DatasetDict
dataset = DatasetDict({
    'member': Dataset.from_dict({'data': train_data}),
    'nonmember': Dataset.from_dict({'data': test_data}),
})

print(dataset)