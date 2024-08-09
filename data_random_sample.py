import os
import torch
import random
from datasets import DatasetDict, Dataset

dataset_name = "arxiv"
train_folder = "/model/pile/by_dataset/"
test_folder = "/model/pile/by_dataset/"

# Step 1: 检测文件夹下有多少train_{dataset_name}_x
train_files = [f for f in os.listdir(train_folder) if f.startswith(f"train_{dataset_name}_")]
test_files = [f for f in os.listdir(test_folder) if f.startswith(f"test_{dataset_name}_")]

# Step 2: 随机抽样多个train_{dataset_name}_x
num_samples = 3 # 假设我们抽样3个文件，这个数字可以根据需要调整
sampled_train_files = random.sample(train_files, num_samples)

# Step 3: 把多个train_{dataset_name}_x合并
merged_train_data = []
for file in sampled_train_files:
    train_dataset = torch.load(os.path.join(train_folder, file))
    merged_train_data.extend(train_dataset)

# Step 4: 在合并train_{dataset_name}_x中，随机采样20000个样本
if len(merged_train_data) > 20000:
    merged_train_data = random.sample(merged_train_data, 20000)

# Step 5: 处理test数据
if len(test_files) > 1:
    # 随机抽样多个test_{dataset_name}_x并合并
    num_samples = 3  # 假设我们抽样3个文件，这个数字可以根据需要调整
    sampled_test_files = random.sample(test_files, num_samples)

    merged_test_data = []
    for file in sampled_test_files:
        test_dataset = torch.load(os.path.join(test_folder, file))
        merged_test_data.extend(test_dataset)

    # 随机采样20000个样本
    if len(merged_test_data) > 20000:
        merged_test_data = random.sample(merged_test_data, 20000)
else:
    # 直接对test_{dataset_name}_0随机采样20000个样本
    test_dataset = torch.load(os.path.join(test_folder, f"test_{dataset_name}_0.pt"))
    if len(test_dataset) > 20000:
        merged_test_data = random.sample(test_dataset, 20000)
    else:
        merged_test_data = test_dataset

# 创建DatasetDict
dataset = DatasetDict({
    'member': Dataset.from_dict({'data': merged_train_data}),
    'nonmember': Dataset.from_dict({'data': merged_test_data}),
})

print(dataset)