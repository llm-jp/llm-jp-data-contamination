from scipy.stats import ks_2samp
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

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

shared_datasets = ['FreeLaw',  'Pile-CC',  'Wikipedia (en)']
method_list = ["mink_plus", "prob", "grad"]
model_sizes = ["12b"]

# 计算并打印数据集和长度之间的关系
for temp in [["absolute", "untruncated"]]:
    results = {method_name: {dataset_name: [] for dataset_name in shared_datasets} for method_name in method_list}
    split = temp[0]
    truncated = temp[1]
    if split == "relative":
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
                            f"mia_dataset_results_{dataset_idx}/{dataset_name}/{split}/{truncated}/{min_len}_{model_size}_{method_name}_dict.pkl",
                            "rb"))

                        if method_name == "refer":
                            residual_dict = {dataset_name: {"member": [], "nonmember": []}}
                            loss_dict = pickle.load(open(
                                f"mia_dataset_results_{dataset_idx}/{dataset_name}/{split}/{truncated}/{min_len}_{model_size}_loss_dict.pkl",
                                "rb"))
                            refer_dict = pickle.load(open(
                                f"mia_dataset_results_{dataset_idx}/{dataset_name}/{split}/{truncated}/{min_len}_{model_size}_refer_dict.pkl",
                                "rb"))

                            member_set = ["member", "nonmember"]

                            for member in member_set:
                                residual_dict[dataset_name][member] = [
                                    loss_dict[dataset_name][member][i] - refer_dict[dataset_name][member][i]
                                    for i in range(len(loss_dict[dataset_name][member]))
                                ]
                            value_dict = residual_dict

                        y_true += [0] * len(value_dict[dataset_name]["member"])
                        y_true += [1] * len(value_dict[dataset_name]["nonmember"])
                        y_scores += value_dict[dataset_name]["member"]
                        y_scores += value_dict[dataset_name]["nonmember"]

                    auc_score = roc_auc_score(y_true, y_scores)
                    temp_score.append(auc_score)

                avg_score = sum(temp_score) / len(temp_score)
                if avg_score > 0.5:
                    results[method_name][dataset_name].append(avg_score)
                else:
                    results[method_name][dataset_name].append(1 - avg_score)

    # 绘图，每个方法一个子图
    num_methods = len(method_list)
    fig, axes = plt.subplots(num_methods, 1, figsize=(12, 8 * num_methods))

    for method_idx, method_name in enumerate(method_list):
        ax = axes[method_idx]
        for dataset_name in shared_datasets:
            # 绘制散点图
            #ax.scatter(x_vals, results[method_name][dataset_name], label=dataset_name)

            # 使用LOESS平滑曲线
            smoothed = sm.nonparametric.lowess(results[method_name][dataset_name], x_vals, frac=0.3)
            ax.plot(x_vals, [y for x, y in smoothed], label=dataset_name)

        ax.set_xlabel('min_len')
        ax.set_ylabel('ROC-AUC Score')
        ax.set_title(f'{method_name} - ROC-AUC Score ({split} - {truncated})')
        ax.legend(loc='best', fontsize='small')

    # 优化布局和节省空间
    plt.tight_layout(pad=1.0)
    plt.savefig(f'results_plot_{split}_{truncated}.png', dpi=300)
    plt.show()