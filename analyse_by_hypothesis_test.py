from scipy.stats import ks_2samp
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import pandas as pd
from statsmodels.regression.linear_model import OLS

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

dataset_idx = 0
split = "absolute"
truncated = "truncated"
if split == "relative":
    length_list = np.arange(0, 100, 10)
else:
    length_list = np.arange(0, 1000, 100)
if split == "absolute" and truncated == "untruncated":
    dataset_names = ['Wikipedia (en)',  "StackExchange", "Pile-CC", "Github", "FreeLaw"]
elif split == "absolute" and truncated == "truncated":
    dataset_names = ['Wikipedia (en)', "StackExchange", 'PubMed Central', "Pile-CC", "HackerNews",
                   "Github", "FreeLaw", "EuroParl",'DM Mathematics',"ArXiv"]
elif split == "relative":
    dataset_names = ["Wikipedia (en)",  "StackExchange",'PubMed Central', "Pile-CC", "NIH ExPorter", "HackerNews",
                   "Github", "FreeLaw", "Enron Emails",  "DM Mathematics", "ArXiv"]

member_set = ["member", "nonmember"]
model_sizes = ["1b", "2.8b", "6.9b", "12b"]
for model_size in model_sizes:
    count = 0
    differentiable_counter = 0
    for dataset_idx in range(3):
        for min_len in length_list:
            for dataset_name in dataset_names:
                method_list = ["loss", "prob", "ppl", "mink_plus", "zlib", "refer", "grad"]
                for idx, method_name in enumerate(method_list):
                    value_dict = pickle.load(open(
                        f"mia_dataset_results_{dataset_idx}/{dataset_name}/{split}/{truncated}/{min_len}_{model_size}_{method_name}_dict.pkl",
                        "rb"))
                    if method_name == "refer":
                        residual_dict = {}
                        residual_dict[dataset_name] = {"member": [], "nonmember": []}
                        loss_dict = pickle.load(open(
                            f"mia_dataset_results_{dataset_idx}/{dataset_name}/{split}/{truncated}/{min_len}_{model_size}_loss_dict.pkl",
                            "rb"))
                        refer_dict = pickle.load(open(
                            f"mia_dataset_results_{dataset_idx}/{dataset_name}/{split}/{truncated}/{min_len}_{model_size}_refer_dict.pkl",
                            "rb"))
                        for member in member_set:
                            residual_dict[dataset_name][member] = [
                                loss_dict[dataset_name][member][i] - refer_dict[dataset_name][member][i]
                                for i in range(len(loss_dict[dataset_name][member]))]
                        value_dict = residual_dict
                    ks_matrix, ks_p_value_matrix = ks_hypothesis(value_dict, dataset_name, split_set=member_set)
                    #print(ks_matrix[0][1])
                    count += 1
                    if ks_p_value_matrix[0][1] <= 0.05:
                        differentiable_counter += 1
    print(model_size)
    print(differentiable_counter/count)

shared_datasets =['FreeLaw', 'Github', 'Pile-CC', 'StackExchange', 'Wikipedia (en)']
method_list = ["loss", "prob", "ppl", "mink_plus", "zlib", "refer", "grad"]
model_sizes = ["1b", "2.8b", "6.9b", "12b"]
for method_name in method_list:
    for model_size in model_sizes:
        count = 0
        differentiable_counter = 0
        for dataset_idx in range(3):
            for temp in [["relative", "truncated"], ["absolute", "truncated"], ["absolute", "untruncated"]]:
                split = temp[0]
                truncated = temp[1]
                if temp[0] == "relative":
                    length_list = np.arange(0, 100, 10)
                else:
                    length_list = np.arange(0, 1000, 100)
                for min_len in length_list:
                    for dataset_name in shared_datasets:
                        value_dict = pickle.load(open(
                            f"mia_dataset_results_{dataset_idx}/{dataset_name}/{split}/{truncated}/{min_len}_{model_size}_{method_name}_dict.pkl",
                            "rb"))
                        if method_name == "refer":
                            residual_dict = {}
                            residual_dict[dataset_name] = {"member": [], "nonmember": []}
                            loss_dict = pickle.load(open(
                                f"mia_dataset_results_{dataset_idx}/{dataset_name}/{split}/{truncated}/{min_len}_{model_size}_loss_dict.pkl",
                                "rb"))
                            refer_dict = pickle.load(open(
                                f"mia_dataset_results_{dataset_idx}/{dataset_name}/{split}/{truncated}/{min_len}_{model_size}_refer_dict.pkl",
                                "rb"))
                            for member in member_set:
                                residual_dict[dataset_name][member] = [
                                    loss_dict[dataset_name][member][i] - refer_dict[dataset_name][member][i]
                                    for i in range(len(loss_dict[dataset_name][member]))]
                            value_dict = residual_dict
                        ks_matrix, ks_p_value_matrix = ks_hypothesis(value_dict, dataset_name, split_set=member_set)
                        #print(ks_matrix[0][1])
                        count += 1
                        if ks_p_value_matrix[0][1] <= 0.05:
                            differentiable_counter += 1
        print(method_name)
        print(model_size)
        print(differentiable_counter/count)


shared_datasets =['FreeLaw', 'Github', 'Pile-CC', 'StackExchange', 'Wikipedia (en)']
method_list = ["loss", "prob", "ppl", "mink_plus", "zlib", "refer", "grad"]
model_sizes = ["1b", "2.8b", "6.9b", "12b"]
for temp in [["relative", "truncated"], ["absolute", "truncated"], ["absolute", "untruncated"]]:
    split = temp[0]
    truncated = temp[1]
    if temp[0] == "relative":
        length_list = np.arange(0, 100, 10)
    else:
        length_list = np.arange(0, 1000, 100)
    for model_size in model_sizes:
        for min_len in length_list:
            count = 0
            differentiable_counter = 0
            for dataset_idx in range(3):
                for method_name in method_list:
                    for dataset_name in shared_datasets:
                        value_dict = pickle.load(open(
                            f"mia_dataset_results_{dataset_idx}/{dataset_name}/{split}/{truncated}/{min_len}_{model_size}_{method_name}_dict.pkl",
                            "rb"))
                        if method_name == "refer":
                            residual_dict = {}
                            residual_dict[dataset_name] = {"member": [], "nonmember": []}
                            loss_dict = pickle.load(open(
                                f"mia_dataset_results_{dataset_idx}/{dataset_name}/{split}/{truncated}/{min_len}_{model_size}_loss_dict.pkl",
                                "rb"))
                            refer_dict = pickle.load(open(
                                f"mia_dataset_results_{dataset_idx}/{dataset_name}/{split}/{truncated}/{min_len}_{model_size}_refer_dict.pkl",
                                "rb"))
                            for member in member_set:
                                residual_dict[dataset_name][member] = [
                                    loss_dict[dataset_name][member][i] - refer_dict[dataset_name][member][i]
                                    for i in range(len(loss_dict[dataset_name][member]))]
                            value_dict = residual_dict
                        ks_matrix, ks_p_value_matrix = ks_hypothesis(value_dict, dataset_name, split_set=member_set)
                        #print(ks_matrix[0][1])
                        count += 1
                        if ks_p_value_matrix[0][1] <= 0.05:
                            differentiable_counter += 1
            print(temp)
            print(min_len)
            print(model_size)
            print(differentiable_counter/count)

hared_datasets = ['FreeLaw', 'Pile-CC', 'Wikipedia (en)', 'Github', 'StackExchange']
method_list = ['mink_plus', 'prob', 'ppl']
model_sizes = ['12b']

# 定义一个存储不同 split 结果的字典
results_split = {'absolute_untruncated': {}, 'absolute_truncated': {}, 'relative_truncated': {}}

split_conditions = [['absolute', 'untruncated'], ['absolute', 'truncated'], ['relative', 'truncated']]

# 计算并打印数据集和长度之间的关系
for condition in split_conditions:
    split = condition[0]
    truncated = condition[1]
    split_key = f'{split}_{truncated}'

    results = {method_name: {dataset_name: [] for dataset_name in shared_datasets} for method_name in method_list}
    results_split[split_key] = results

    if split == 'relative':
        length_list = np.arange(0, 100, 10)
    else:
        length_list = np.arange(0, 1000, 100)

    x_vals = length_list

    for dataset_name in shared_datasets:
        for min_len in length_list:
            for method_name in method_list:
                y_true = []
                y_scores = []

                for model_size in model_sizes:
                    temp_score = []

                    for dataset_idx in range(3):
                        # 加载值字典
                        value_dict = pickle.load(open(
                            f'mia_dataset_results_{dataset_idx}/{dataset_name}/{split}/{truncated}/{min_len}_{model_size}_{method_name}_dict.pkl',
                            'rb'))

                        if method_name == 'refer':
                            residual_dict = {dataset_name: {'member': [], 'nonmember': []}}
                            loss_dict = pickle.load(open(
                                f'mia_dataset_results_{dataset_idx}/{dataset_name}/{split}/{truncated}/{min_len}_{model_size}_loss_dict.pkl',
                                'rb'))
                            refer_dict = pickle.load(open(
                                f'mia_dataset_results_{dataset_idx}/{dataset_name}/{split}/{truncated}/{min_len}_{model_size}_refer_dict.pkl',
                                'rb'))

                            member_set = ['member', 'nonmember']

                            for member in member_set:
                                residual_dict[dataset_name][member] = [
                                    loss_dict[dataset_name][member][i] - refer_dict[dataset_name][member][i]
                                    for i in range(len(loss_dict[dataset_name][member]))
                                ]
                            value_dict = residual_dict

                        y_true += [0] * len(value_dict[dataset_name]['member'])
                        y_true += [1] * len(value_dict[dataset_name]['nonmember'])
                        y_scores += value_dict[dataset_name]['member']
                        y_scores += value_dict[dataset_name]['nonmember']

                    auc_score = roc_auc_score(y_true, y_scores)
                    temp_score.append(auc_score)

                avg_score = sum(temp_score) / len(temp_score)
                if avg_score > 0.5:
                    results[method_name][dataset_name].append((min_len, avg_score))
                else:
                    results[method_name][dataset_name].append((min_len, 1 - avg_score))

# 计算每个数据集在每个方法上随着length增加的变化程度，并分开表格
summary_tables = []

for split_key, results in results_split.items():
    summary_table = []

    for method_name in method_list:
        for dataset_name in shared_datasets:
            lengths, scores = zip(*results[method_name][dataset_name])
            X = sm.add_constant(np.array(lengths) / 100)  # 将length标准化到单位100
            model = OLS(scores, X).fit()
            change_rate = model.params[1] * 100  # 线性回归的斜率乘以100，即变化率
            max_length = lengths[np.argmax(scores)]
            max_score = max(scores)
            summary_table.append([method_name, dataset_name, max_length, max_score, change_rate])

    df = pd.DataFrame(summary_table, columns=['Method', 'Dataset', 'Max Length', 'Max ROC-AUC Score', 'Change Rate'])
    summary_tables.append((split_key, df))

# 打印结果表格
for split_key, df in summary_tables:
    print(f'Results for {split_key}:')
    print(df)
    print('\n')