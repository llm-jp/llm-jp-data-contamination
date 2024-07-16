import pickle
import numpy as np
#from utils import remove_outliers
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
def calculate_statistics(data):
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'var': np.var(data),
        'kurtosis': kurtosis(data)
    }
stats_data = {
    "mean": {},
    "var": {},
    "std": {},
    "kur": {},
}
def remove_outliers(data, m=2):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    # 找到大于均值 + m * std 和小于均值 - m * std 的离群值
    outliers_high = data > mean + m * std
    outliers_low = data < mean - m * std
    outliers = outliers_high | outliers_low
    # 计算没有离群值的平均值
    mean_without_outliers = np.mean(data[~outliers])
    # 用没有离群值的平均值替换离群值
    data[outliers] = mean_without_outliers
    return data.tolist()
model_size_list = ["160m", "1b", "2.8b"]
dataset_names = ["ArXiv", "DM Mathematics","FreeLaw", "Github", "HackerNews", "NIH ExPorter",
                "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                 "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA"]
statistics = {dataset_name: {model_size: {set_name: {} for set_name in ["train", "test", "valid"]} for model_size in model_size_list} for dataset_name in dataset_names}

for dataset_name in dataset_names:
    print(dataset_name)
    for model_size in model_size_list:
        loss_dict = pickle.load(open(f"feature_result/{dataset_name}_{model_size}_loss_dict.pkl", "rb"))
        prob_dict = pickle.load(open(f"feature_result/{dataset_name}_{model_size}_prob_dict.pkl", "rb"))
        ppl_dict = pickle.load(open(f"feature_result/{dataset_name}_{model_size}_ppl_dict.pkl", "rb"))
        mink_plus_dict = pickle.load(open(f"feature_result/{dataset_name}_{model_size}_mink_plus_dict.pkl", "rb"))
        zlib_dict = pickle.load(open(f"feature_result/{dataset_name}_{model_size}_zlib_dict.pkl", "rb"))
        for data_dict, metric in zip([loss_dict, prob_dict, ppl_dict, mink_plus_dict, zlib_dict],
                                     ["loss", "probability", "ppl", "mink_plus", "zlib"]):
            for set_name in ["train", "test", "valid"]:
                data = np.array(data_dict[dataset_name][set_name])
                data = remove_outliers(data[~np.isnan(data)])
                statistics[dataset_name][model_size][set_name][metric] = calculate_statistics(data)

            # Now plot each statistic separately
for dataset_name in dataset_names:
    for metric in ["loss", "probability", "ppl", "mink_plus", "zlib"]:
        plt.figure(figsize=(12, 8))

        for stat in ["mean", "std", "var", "kurtosis"]:
            for set_name in ["train", "test", "valid"]:
                stat_values = []
                for model_size in model_size_list:
                    stat_values.append(statistics[dataset_name][model_size][set_name][metric][stat])
                stat_values = np.array(stat_values)
                plt.plot(model_size_list, stat_values, label=f'{set_name} {stat}')

        plt.title(f'{dataset_name} {metric} statistics')
        plt.legend(loc='best')
        plt.xlabel('Model size')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()
