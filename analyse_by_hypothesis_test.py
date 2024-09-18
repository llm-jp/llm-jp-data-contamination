from scipy.stats import ks_2samp
import numpy as np
import pickle


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

shared_datasets =['FreeLaw', 'Github', 'Pile-CC', 'StackExchange', 'Wikipedia (en)']
method_list = ["loss", "prob", "ppl", "mink_plus", "zlib", "refer", "grad"]
#model_sizes = ["1b", "2.8b", "6.9b", "12b"]
model_sizes = ["12b"]
# 计算并打印数据集和长度之间的关系
#["absolute", "truncated"], ["absolute", "untruncated"]
for temp in [["relative", "truncated"]]:
    split = temp[0]
    truncated = temp[1]
    if split == "relative":
        length_list = np.arange(0, 100, 10)
    else:
        length_list = np.arange(0, 1000, 100)
    for dataset_name in shared_datasets:
        for min_len in length_list:
            count = 0
            differentiable_counter = 0
            for model_size in model_sizes:
                for dataset_idx in range(3):
                    for method_name in method_list:
                        # 加载值字典
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
                        # print(ks_matrix[0][1])
                        count += 1
                        if ks_p_value_matrix[0][1] <= 0.05:
                            differentiable_counter += 1
            print(temp)
            print(dataset_name)
            print(min_len)
            print(differentiable_counter/count)