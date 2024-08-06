import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

#dataset_names = ["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
#                      "pubmed_central", "wikipedia_(en)", "full_pile"]
dataset_names = [ "github", "pile_cc", "full_pile", "WikiMIA64", "WikiMIA128", "WikiMIA256", "WikiMIAall"]
model_size = "12b"
results = []
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
                results.append([dataset_name, model_size, dict_name, label, s])

df = pd.DataFrame(results, columns=['dataset', 'model_size', 'feature', 'label', 'score'])
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

        # 根据8:2比例拆分数据集
        X_train, X_test, y_train, y_test = train_test_split(df_feature['score'], df_feature['label'], test_size=0.2,
                                                            random_state=42)

        # 计算训练集的ROC曲线
        fpr, tpr, thresholds = roc_curve(y_train, X_train)

        # 计算AUC值
        auc = roc_auc_score(y_train, X_train)
        print(f'AUC for dataset {dataset_name}, feature {feature}: {auc}')

        # 找到最佳阈值 - Youden's Index
        J = tpr - fpr
        best_ix = np.argmax(J)
        best_threshold = thresholds[best_ix]

        # 在训练集上评估
        y_train_pred = X_train >= best_threshold  # 将得分与阈值比较得到预测标签

        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)

        # 在测试集上评估
        y_test_pred = X_test >= best_threshold  # 将得分与阈值比较得到预测标签

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)

        # 存储评估结果
        feature_results[feature] = {
            'best_threshold': best_threshold,
            'train_accuracy': train_accuracy,
            'train_recall': train_recall,
            'train_precision': train_precision,
            'test_accuracy': test_accuracy,
            'test_recall': test_recall,
            'test_precision': test_precision
        }

        print(f'Best Threshold for dataset {dataset_name}, feature {feature}: {best_threshold} (J = {J[best_ix]})')
        print(f'Training set - Accuracy: {train_accuracy}, Recall: {train_recall}, Precision: {train_precision}')
        print(f'Test set - Accuracy: {test_accuracy}, Recall: {test_recall}, Precision: {test_precision}')

        # 可视化ROC曲线
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for dataset {dataset_name}, feature: {feature}')
        plt.legend(loc="lower right")
        plt.show()

    evaluation_results[dataset_name] = feature_results

# 打印所有数据集和特征的评估结果
print("Evaluation results for each dataset and feature:")
for dataset_name, feature_results in evaluation_results.items():
    print(f"Dataset: {dataset_name}")
    for feature, metrics in feature_results.items():
        print(f"  Feature: {feature}")
        for metric, value in metrics.items():
            print(f"    {metric}: {value}")


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
            'test_precision': metrics['test_precision']
        }
        results_list.append(entry)

results_df = pd.DataFrame(results_list)

# 美化表格显示
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
            'test_precision': metrics['test_precision']
        }
        results_list.append(entry)

results_df = pd.DataFrame(results_list)

# 打印DataFrame
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