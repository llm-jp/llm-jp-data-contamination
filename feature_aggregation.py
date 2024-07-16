from datasets import load_dataset
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
import pickle
from itertools import islice
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch.nn.functional as F
from scipy.stats import entropy, ks_2samp, kurtosis, wasserstein_distance
import argparse
import random
import seaborn as sns
import zlib
import os
from utils import *

dataset_name = "Pile-CC"
model_size = "160m"
loss_dict = pickle.load(open(f"feature_result/{dataset_name}_{model_size}_loss_dict.pkl", "rb"))
prob_dict = pickle.load(open(f"feature_result/{dataset_name}_{model_size}_prob_dict.pkl", "rb"))
ppl_dict = pickle.load(open(f"feature_result/{dataset_name}_{model_size}_ppl_dict.pkl", "rb"))
mink_plus_dict = pickle.load(open(f"feature_result/{dataset_name}_{model_size}_mink_plus_dict.pkl", "rb"))
zlib_dict = pickle.load(open(f"feature_result/{dataset_name}_{model_size}_zlib_dict.pkl", "rb"))

aggregated_train = []
aggregated_test = []
aggregated_val = []
for dict in [loss_dict, prob_dict, ppl_dict, mink_plus_dict, zlib_dict]:
    for set_name in ["train", "test", "valid"]:
        data = np.array(dict[dataset_name][set_name])
        mean1, std1 = np.mean(data), np.std(data)
        normalized_value = (dict[dataset_name][set_name] - mean1) / std1
        if set_name == "train":
            aggregated_train.extend(normalized_value.tolist())
        elif set_name == "test":
            aggregated_test.extend(normalized_value.tolist())
        else:
            aggregated_val.extend(normalized_value.tolist())

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# 去除离群值
train_no_outliers = remove_outliers(aggregated_train)
test_no_outliers = remove_outliers(aggregated_test)
val_no_outliers = remove_outliers(aggregated_val)

# 第一个子图：绘制KDE
sns.kdeplot(train_no_outliers, ax=axs[0], label="train", alpha=0.5, bw_adjust=0.1, shade=True)
sns.kdeplot(test_no_outliers, ax=axs[0], label="test", alpha=0.5, bw_adjust=0.1, shade=True)
sns.kdeplot(val_no_outliers, ax=axs[0], label="val", alpha=0.5, bw_adjust=0.1, shade=True)
axs[0].set_title(f'{dataset_name} KDE PDF at {model_size} model')
axs[0].set_xlabel("Normalized Value")
axs[0].set_ylabel('KDE PDF Value')
axs[0].legend()

# 第二个子图：绘制直方图
bins = np.linspace(-2.5, 2.5, 300)  # 可以根据数据范围调整
sns.histplot(train_no_outliers, ax=axs[1], label="train", bins=bins, alpha=0.5, stat="density", kde=False)
sns.histplot(test_no_outliers, ax=axs[1], label="test", bins=bins, alpha=0.5, stat="density", kde=False)
sns.histplot(val_no_outliers, ax=axs[1], label="val", bins=bins, alpha=0.5, stat="density", kde=False)
axs[1].set_title(f'{dataset_name} Histogram PDF at {model_size} model')
axs[1].set_xlabel("Normalized Value")
axs[1].set_ylabel('Histogram PDF Value')
axs[1].legend()

# 调整布局并显示图形
plt.tight_layout()
plt.show()


