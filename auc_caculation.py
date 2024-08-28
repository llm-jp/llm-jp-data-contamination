import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, matthews_corrcoef
from sklearn.model_selection import train_test_split


# Method definitions remain the same here

# 定义计算各个评估方法的阈值函数
def percentile_method(y_true, scores, percentile=95):
    threshold = np.percentile(scores, percentile)
    return threshold


def gmean_method(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    gmeans = np.sqrt(tpr * (1 - fpr))
    idx = np.argmax(gmeans)
    return thresholds[idx]


def roc_distance_method(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
    idx = np.argmin(distances)
    return thresholds[idx]


def f1_score_max_method(y_true, scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    idx = np.argmax(f1_scores)
    return thresholds[idx]


def mcc_method(y_true, scores):
    fprs, tprs, thresholds = roc_curve(y_true, scores)
    mccs = [(matthews_corrcoef(y_true, scores >= thr), thr) for thr in thresholds]
    best_thr = max(mccs, key=lambda x: x[0])[1]
    return best_thr


def precision_recall_cutoff(y_true, scores, target_recall=0.8):
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    left_index = np.where(recalls >= target_recall)[0]
    idx = np.argmax(precisions[left_index])
    return thresholds[left_index[idx]]


def decide_threshold_direction(y_true, scores, threshold):
    positive_scores = scores[y_true == 1]
    negative_scores = scores[y_true == 0]
    pos_mean = np.mean(positive_scores)
    neg_mean = np.mean(negative_scores)
    return '<=' if pos_mean < neg_mean else '>='

#truncated
#dataset_names = ['Wikipedia (en)', "StackExchange", 'PubMed Central', "Pile-CC", "HackerNews",
#                    "Github", "FreeLaw", "EuroParl", 'DM Mathematics', "ArXiv", ]
split = "relative"
truncate_status = "truncated"
if split == "relative":
    length_values = np.arange(0, 100, 10)
else:
    length_values = np.arange(0, 1000, 100)
#untruncated
#dataset_names = ['Wikipedia (en)', "USPTO Backgrounds", "StackExchange", "Pile-CC", "Github", "FreeLaw"]
#relative
dataset_names = ["Wikipedia (en)", "StackExchange", 'PubMed Central', "Pile-CC", "NIH ExPorter", "HackerNews",
                    "Github", "FreeLaw", "Enron Emails", "DM Mathematics", "ArXiv"]
# 随机种子列表
#dataset_names = ["github", "pile_cc", "full_pile", "WikiMIA64", "WikiMIA128", "WikiMIA256", "WikiMIAall"]
model_sizes = ["410m", "1b", "2.8b", "6.9b"]
results = []
auc_results = []
for model_size in model_sizes:
    for length in length_values:
        for dataset_name in dataset_names:
            base_path = f"mia_dataset_results/{dataset_name}/{split}/{truncate_status}/{length}_{model_size}_"
            loss_dict = pickle.load(open(base_path + "loss_dict.pkl", "rb"))
            prob_dict = pickle.load(open(base_path + "prob_dict.pkl", "rb"))
            ppl_dict = pickle.load(open(base_path + "ppl_dict.pkl", "rb"))
            mink_plus_dict = pickle.load(open(base_path + "mink_plus_dict.pkl", "rb"))
            zlib_dict = pickle.load(open(base_path + "zlib_dict.pkl", "rb"))
            grad_dict = pickle.load(open(base_path + "grad_dict.pkl", "rb"))
            refer_dict = pickle.load(open(base_path + "refer_dict.pkl", "rb"))

            dict_list = [loss_dict, prob_dict, ppl_dict, mink_plus_dict, zlib_dict, grad_dict, refer_dict]
            dict_names = ["loss", "prob", "ppl", "mink_plus", "zlib", "grad", "refer"]

            # Process scores for each dictionary
            for dict_name, d in zip(dict_names, dict_list):
                member_score = np.array(d[dataset_name]["member"])
                nonmember_score = np.array(d[dataset_name]["nonmember"])

                # Add data to results list
                for score, label in [(member_score, 'member'), (nonmember_score, 'nonmember')]:
                    for s in score:
                        results.append([dataset_name, model_size, dict_name, label, s, length])

# Create DataFrame from results
df = pd.DataFrame(results, columns=['dataset', 'model_size', 'feature', 'label', 'score', 'length'])
df['label'] = df['label'].apply(lambda x: 1 if x == 'member' else 0)  # Convert label to binary

# Evaluation results
evaluation_results = {}

# Calculate evaluation metrics for each dataset, feature, length and model size
for dataset_name in dataset_names:
    df_dataset = df[df['dataset'] == dataset_name]
    features = df_dataset['feature'].unique()

    for feature in features:
        unique_model_sizes = df_dataset['model_size'].unique()

        for model_size in unique_model_sizes:
            unique_lengths = df_dataset['length'].unique()

            for length in unique_lengths:
                df_feature_length = df_dataset[(df_dataset['feature'] == feature) &
                                               (df_dataset['length'] == length) &
                                               (df_dataset['model_size'] == model_size)]
                overall_true_labels = df_feature_length['label']
                overall_scores = df_feature_length['score']
                try:
                    overall_auc = roc_auc_score(overall_true_labels, overall_scores)
                except ValueError:
                    overall_auc = 0.5

                # If AUC is less than 0.5, invert the scores and recalculate
                if overall_auc < 0.5:
                    overall_auc = 1 - overall_auc
                    overall_scores = -overall_scores

                print(
                    f"Dataset: {dataset_name}, Feature: {feature}, Length: {length}, Model Size: {model_size}, AUC: {overall_auc}")
                auc_results.append([dataset_name, feature, length, model_size, overall_auc])

# Create DataFrame for AUC results
auc_df = pd.DataFrame(auc_results, columns=['dataset', 'feature', 'length', 'model_size', 'auc'])

auc_df.to_csv(f"auc_results_{split}_{truncate_status}_all_length_all_model_size.csv", index=False)