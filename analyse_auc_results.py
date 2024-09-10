import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
truncated_results = pd.read_csv("auc_results_absolute_truncated_all_length_all_model_size.csv")
relative_results = pd.read_csv("auc_results_relative_truncated_all_length_all_model_size.csv")
untruncated_results = pd.read_csv("auc_results_absolute_untruncated_all_length_all_model_size.csv")

# 设置Seaborn样式
sns.set(style="whitegrid")

def draw_by_dataset(csv_results, dataset_name, plot_title):
    """ 绘制特定数据集的AUC值随长度变化的图 """
    df_filtered = csv_results[csv_results["dataset"] == dataset_name]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x="length", y="auc", marker="o")
    plt.title(plot_title, fontsize=16)
    plt.xlabel("Length", fontsize=14)
    plt.ylabel("AUC", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def draw_by_dataset_and_model_size(csv_results, dataset_name, plot_title):
    """ 绘制特定数据集，不同模型大小在同一图中的 AUC 值随长度变化的图 """
    df_filtered = csv_results[csv_results["dataset"] == dataset_name]
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_filtered, x="length", y="auc", hue="model_size", marker="o", palette="viridis")
    plt.title(plot_title, fontsize=16)
    plt.xlabel("Length", fontsize=14)
    plt.ylabel("AUC", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Model Size', fontsize=12, title_fontsize='13')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def draw_by_feature(csv_results, dataset_name, plot_title):
    """ 绘制特定数据集，不同特征在同一图中的 AUC 值随长度变化的图 """
    df_filtered = csv_results[csv_results["dataset"] == dataset_name]
    unique_features = df_filtered['feature'].unique()
    for feature in unique_features:
        df_feature = df_filtered[df_filtered['feature'] == feature]
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df_feature, x="length", y="auc", hue="model_size", marker="o", palette="viridis")
        plt.title(f"{plot_title} - {feature}", fontsize=16)
        plt.xlabel("Length", fontsize=14)
        plt.ylabel("AUC", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Model Size', fontsize=12, title_fontsize='13')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()

def plot_hist_by_category(df, category, title, bins=30, xmin=0.5, xmax=1.0, show_hist=False):
    """ 绘制指定分类的AUC概率密度函数图 """
    plt.figure(figsize=(12, 8))
    sns.set_palette("Set2")
    unique_categories = df[category].unique()
    if len(unique_categories) > 10:
        sns.set_palette(sns.color_palette("husl", len(unique_categories)))
    for i, cat in enumerate(unique_categories):
        subset = df[df[category] == cat]
        data = subset['auc']
        if show_hist:
            sns.histplot(data, bins=bins, kde=True, label=cat, stat='density', element='step', linewidth=1.5)
        else:
            sns.kdeplot(data, label=cat, linewidth=2, linestyle=('-' if i % 2 == 0 else '--'), alpha=0.7)
    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel("AUC", fontsize=16, weight='bold')
    plt.ylabel("Density", fontsize=16, weight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(xmin, xmax)
    plt.ylim(0, None)
    plt.legend(title=category, fontsize=12, title_fontsize='13', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

def plot_pdf_comparison(datasets, dataset_name, feature, title, bins=30, xmin=0.5, xmax=1.0, show_hist=False):
    """ 对比不同数据集在相同特征上的概率密度函数 """
    plt.figure(figsize=(14, 10))
    sns.set_palette("Set2")
    dataset_labels = ["Truncated", "Relative", "Untruncated"]
    for i, (df, label) in enumerate(zip(datasets, dataset_labels)):
        df_filtered = df[(df["dataset"] == dataset_name) & (df["feature"] == feature)]
        if show_hist:
            sns.histplot(df_filtered['auc'], bins=bins, kde=True, label=f"{label}", stat='density', element='step', linewidth=1.5)
        else:
            sns.kdeplot(df_filtered['auc'], label=f"{label}", linewidth=2, linestyle=('-' if i % 2 == 0 else '--'), alpha=0.7)
    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel("AUC", fontsize=16, weight='bold')
    plt.ylabel("Density", fontsize=16, weight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(xmin, xmax)
    plt.ylim(0, None)
    plt.legend(title='Dataset Type', fontsize=12, title_fontsize='13', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

def plot_pdf_comparison_all_methods(datasets, dataset_labels, title, bins=30, xmin=0.5, xmax=1.0, show_hist=False):
    """ 对比不同数据框（方法）的总体概率密度函数 """
    plt.figure(figsize=(14, 10))
    sns.set_palette("Set2")
    for i, (df, label) in enumerate(zip(datasets, dataset_labels)):
        total_auc = pd.Series(dtype='float64')
        common_datasets = set(df['dataset'].unique()).intersection(set(datasets[0]['dataset'].unique()), set(datasets[1]['dataset'].unique()))
        for dataset_name in common_datasets:
            df_filtered = df[df['dataset'] == dataset_name]
            total_auc = total_auc.append(df_filtered['auc'])
        kde = sns.kdeplot(total_auc, label=label, linewidth=2, alpha=0.7 if show_hist else 1.0)
        if not show_hist:
            x, y = kde.get_lines()[-1].get_data()
            plt.fill_between(x, 0, y, alpha=0.2)
    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel("AUC", fontsize=16, weight='bold')
    plt.ylabel("Density", fontsize=16, weight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(xmin, xmax)
    plt.ylim(0, None)
    plt.legend(title='Method', fontsize=12, title_fontsize='13', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

# 使用绘图函数
datasets = [truncated_results, relative_results, untruncated_results]
dataset_labels = ["Truncated", "Relative", "Untruncated"]
shared_datasets = set(relative_results['dataset'].unique()) & set(truncated_results['dataset'].unique()) & set(untruncated_results['dataset'].unique())

# 示例绘图调用
#draw_by_dataset(truncated_results, "Wikipedia (en)", "Absolute AUC for ArXiv")
#draw_by_dataset(relative_results, "Wikipedia (en)", "Absolute AUC for ArXiv")
#draw_by_dataset(untruncated_results, "Wikipedia (en)", "Absolute AUC for ArXiv")

draw_by_dataset_and_model_size(truncated_results, "Pile-CC", "Absolute AUC for Pile-CC by Model Size")
draw_by_dataset_and_model_size(relative_results, "Pile-CC", "Relative AUC for Pile-CC by Model Size")
draw_by_dataset_and_model_size(untruncated_results, "Pile-CC", "Untruncated AUC for Pile-CC by Model Size")

draw_by_feature(truncated_results, "Github", "Absolute AUC for Github by Model Size and Feature")

plot_hist_by_category(relative_results, "dataset", "Probability Density Function of AUC for All Datasets", bins=30, xmin=0.5, xmax=0.6, show_hist=False)
plot_hist_by_category(truncated_results, "dataset", "Probability Density Function of AUC for All Model Sizes", bins=30, xmin=0.5, xmax=0.6, show_hist=False)
plot_hist_by_category(untruncated_results, "dataset", "Probability Density Function of AUC for All Model Sizes", bins=30, xmin=0.5, xmax=0.6, show_hist=False)

plot_pdf_comparison(datasets, "Wikipedia (en)", "loss", "Probability Density Function Comparison", bins=30, xmin=0.5, xmax=0.6, show_hist=False)
plot_pdf_comparison_all_methods(datasets, dataset_labels, "Probability Density Function Comparison of Different Methods", bins=30, xmin=0.5, xmax=0.6, show_hist=False)