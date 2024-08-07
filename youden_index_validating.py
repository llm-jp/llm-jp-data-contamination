import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, precision_recall_curve
from sklearn.model_selection import train_test_split

# 方法定义
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

# 随机种子列表
random_seeds = [42, 52, 62, 72, 82]
dataset_names = ["github", "pile_cc", "full_pile", "WikiMIA64", "WikiMIA128", "WikiMIA256", "WikiMIAall"]
model_size = "12b"

results = []

for seed in random_seeds:
    for dataset_name in dataset_names:
        loss_dict = pickle.load(open(f"feature_result_online/{dataset_name}_{model_size}_loss_dict.pkl", "rb"))
        prob_dict = pickle.load(open(f"feature_result_online/{dataset_name}_{model_size}_prob_dict.pkl", "rb"))
        ppl_dict = pickle.load(open(f"feature_result_online/{dataset_name}_{model_size}_ppl_dict.pkl", "rb"))
        mink_plus_dict = pickle.load(open(f"feature_result_online/{dataset_name}_{model_size}_mink_plus_dict.pkl", "rb"))
        zlib_dict = pickle.load(open(f"feature_result_online/{dataset_name}_{model_size}_zlib_dict.pkl", "rb"))

        dict_list = [loss_dict, prob_dict, ppl_dict, mink_plus_dict, zlib_dict]
        dict_names = ["loss", "prob", "ppl", "mink_plus", "zlib"]

        # 对每个词典中的成员和非成员得分进行处理
        for dict_name, d in zip(dict_names, dict_list):
            member_score = np.array(d[dataset_name]["member"])
            nonmember_score = np.array(d[dataset_name]["nonmember"])

            # 添加数据到结果列表
            for score, label in [(member_score, 'member'), (nonmember_score, 'nonmember')]:
                for s in score:
                    results.append([dataset_name, model_size, dict_name, label, s, seed])

df = pd.DataFrame(results, columns=['dataset', 'model_size', 'feature', 'label', 'score', 'seed'])
df['label'] = df['label'].apply(lambda x: 1 if x == 'member' else 0)  # 将label转为二进制

# 初始化结果字典，存储每个数据集和特征的评估结果
evaluation_results = {}

# 按dataset和feature分类结果
for dataset_name in dataset_names:
    df_dataset = df[df['dataset'] == dataset_name]
    feature_results = {}

    # 遍历每个特征
    features = df_dataset['feature'].unique()

    for feature in features:
        df_feature = df_dataset[df_dataset['feature'] == feature]
        seed_results = []

        for seed in random_seeds:
            df_seed = df_feature[df_feature['seed'] == seed]

            # 根据8:2比例拆分数据集
            X_train, X_test, y_train, y_test = train_test_split(df_seed['score'], df_seed['label'], test_size=0.2, random_state=seed)

            # 使用G-mean方法计算最佳阈值
            best_threshold = gmean_method(y_train, X_train)
            threshold_direction = decide_threshold_direction(y_train, X_train, best_threshold)

            # 在训练集上评估
            if threshold_direction == '>=':
                y_train_pred = np.array(X_train >= best_threshold, dtype=float)
                y_test_pred = np.array(X_test >= best_threshold, dtype=float)
            else:
                y_train_pred = np.array(X_train <= best_threshold, dtype=float)
                y_test_pred = np.array(X_test <= best_threshold, dtype=float)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_recall = recall_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred)

            # 在测试集上评估
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)

            # 存储评估结果
            seed_results.append({
                'best_threshold': best_threshold,
                'train_accuracy': train_accuracy,
                'train_recall': train_recall,
                'train_precision': train_precision,
                'test_accuracy': test_accuracy,
                'test_recall': test_recall,
                'test_precision': test_precision,
                'threshold_direction': threshold_direction
            })

        # 计算平均值
        avg_results = {metric: np.mean([res[metric] for res in seed_results]) for metric in seed_results[0] if
                       metric != 'threshold_direction'}
        directions = [res['threshold_direction'] for res in seed_results]
        if all(d == directions[0] for d in directions):
            avg_results['threshold_direction'] = directions[0]
        else:
            avg_results['threshold_direction'] = 'Inconsistent'

        feature_results[feature] = avg_results

        # 画出nonmember和member分布图，并标记阈值
        nonmember_scores = df_feature[df_feature['label'] == 0]['score']
        member_scores = df_feature[df_feature['label'] == 1]['score']

        plt.figure(figsize=(10, 6))
        plt.hist(nonmember_scores, bins=50, alpha=0.5, label='Non-member', color='blue')
        plt.hist(member_scores, bins=50, alpha=0.5, label='Member', color='red')

        if avg_results['threshold_direction'] == '>=':
            plt.axvline(avg_results['best_threshold'], color='green', linestyle='dashed', linewidth=2, label=f'Threshold: {avg_results["best_threshold"]:.2f} (>=)')
        else:
            plt.axvline(avg_results['best_threshold'], color='green', linestyle='dashed', linewidth=2, label=f'Threshold: {avg_results["best_threshold"]:.2f} (<=)')

        plt.title(f'Distribution of Non-members and Members\nDataset: {dataset_name}, Feature: {feature}')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

    evaluation_results[dataset_name] = feature_results

# 打印所有数据集和特征的评估结果
print("Evaluation results for each dataset and feature (averaged over seeds):")
for dataset_name, feature_results in evaluation_results.items():
    print(f"Dataset: {dataset_name}")
    for feature, metrics in feature_results.items():
        print(f"  Feature: {feature}")
        for metric, value in metrics.items():
            print(f"    {metric}: {value}")

# 保存结果到DataFrame
results_list = []
for dataset_name, features_results in evaluation_results.items():
    for feature, metrics in features_results.items():
        entry = {
            'dataset': dataset_name,
            'feature': feature,
            'best_threshold': metrics['best_threshold'],
            'train_accuracy': metrics['train_accuracy'],
            'train_recall': metrics['train_recall'],
            'train_precision': metrics['train_precision'],
            'test_accuracy': metrics['test_accuracy'],
            'test_recall': metrics['test_recall'],
            'test_precision': metrics['test_precision'],
            'threshold_direction': metrics['threshold_direction']
        }
        results_list.append(entry)

results_df = pd.DataFrame(results_list)

# 美化表格显示
print(results_df)

# 将数值格式化为保留两位小数
formatters = {column: "{:.2f}".format for column in results_df.columns if results_df[column].dtype == 'float'}

# 转换为 LaTeX 格式并保存到文件
latex_file_path = 'evaluation_results.tex'
with open(latex_file_path, 'w') as f:
    f.write(results_df.to_latex(index=False, formatters=formatters))
print(f'Results have been saved to {latex_file_path}')

# 可视化 - 使用Bar Chart比较训练集和测试集的准确率
fig, ax = plt.subplots(figsize=(10, 6))
train_accuracies = results_df.pivot(index='dataset', columns='feature', values='train_accuracy')
test_accuracies = results_df.pivot(index='dataset', columns='feature', values='test_accuracy')

train_accuracies.plot(kind='bar', ax=ax, position=0, color='blue', width=0.4, align='center', label='Train Accuracy')
test_accuracies.plot(kind='bar', ax=ax, position=1, color='orange', width=0.4, align='edge', label='Test Accuracy')

plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('Training vs Testing Accuracy for Different Datasets and Features')
plt.legend(loc='best')
plt.show()