import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加载并合并数据
dataset_idxs = [0, 1, 2]
truncated_results_list = []
relative_results_list = []
untruncated_results_list = []

for idx in dataset_idxs:
    df_truncated = pd.read_csv(f"auc_results_absolute_truncated_all_length_all_model_size_{idx}.csv")
    df_truncated['seed'] = idx  # 添加种子列
    truncated_results_list.append(df_truncated)

    df_relative = pd.read_csv(f"auc_results_relative_truncated_all_length_all_model_size_{idx}.csv")
    df_relative['seed'] = idx
    relative_results_list.append(df_relative)

    df_untruncated = pd.read_csv(f"auc_results_absolute_untruncated_all_length_all_model_size_{idx}.csv")
    df_untruncated['seed'] = idx
    untruncated_results_list.append(df_untruncated)

# 合并数据框
truncated_results = pd.concat(truncated_results_list, ignore_index=True)
relative_results = pd.concat(relative_results_list, ignore_index=True)
untruncated_results = pd.concat(untruncated_results_list, ignore_index=True)

# 设置Seaborn样式
sns.set(style="whitegrid")


# ----------------------------------------------------------------------------------------------------------------------
# 定义绘图函数
def draw_by_dataset(csv_results, dataset_name, plot_title):
    """
    绘制特定数据集的AUC值随长度变化的图，计算平均值和方差，绘制单个结果和方差区域
    """
    df_filtered = csv_results[csv_results["dataset"] == dataset_name]
    plt.figure(figsize=(10, 6))

    # 绘制单个结果（不同种子）
    sns.lineplot(data=df_filtered, x="length", y="auc", units="seed", estimator=None,
                 color='blue', linewidth=1, alpha=0.3, legend=False)

    # 按照 length 进行分组，计算平均值和标准差
    grouped = df_filtered.groupby(['length'], as_index=False).agg({'auc': ['mean', 'std']})
    grouped.columns = ['length', 'mean_auc', 'std_auc']

    # 绘制平均曲线
    sns.lineplot(data=grouped, x="length", y="mean_auc", color='blue', marker="o")

    # 使用 fill_between 绘制方差区域
    lengths = grouped['length']
    mean_auc = grouped['mean_auc']
    std_auc = grouped['std_auc']

    plt.fill_between(lengths, mean_auc - std_auc, mean_auc + std_auc, color='blue', alpha=0.2)

    plt.title(plot_title, fontsize=16)
    plt.xlabel("Length", fontsize=14)
    plt.ylabel("AUC", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


def plot_pdf_comparison_shared_model_size(datasets, model_sizes, title, bins=30, xmin=0.5, xmax=1.0, show_hist=False):
    """ 对比不同方法在共有模型大小上的概率密度函数，并展示均值和方差区域 """
    plt.figure(figsize=(14, 10))
    sns.set_palette("Set2")
    method_labels = ["Truncated", "Relative", "Untruncated"]
    colors = sns.color_palette("Set2", len(model_sizes))

    for j, model_size in enumerate(model_sizes):
        color = colors[j % len(colors)]

        data_by_method = []

        for df in datasets:
            df_filtered = df[df['model_size'] == model_size]

            # 获取属于特定模型大小的所有种子
            data_by_seed = [df_filtered[df_filtered['seed'] == seed]['auc'] for seed in df_filtered['seed'].unique()]

            # 将所有种子的 AUC 数据汇总
            all_data = np.concatenate(data_by_seed)
            data_by_method.append(all_data)

        # 合并所有方法的 AUC 数据，计算总的均值和方差
        total_data = np.concatenate(data_by_method)
        mean_auc = total_data.mean()
        std_auc = total_data.std()
        sns.kdeplot(total_data, label=f"{model_size}", color=color, linewidth=2, alpha=0.7)

        # 绘制均值的竖直线
        plt.axvline(mean_auc, color=color, linestyle='-', linewidth=1.5, alpha=0.7)

        # 使用 fill_between 填充均值 ± 标准差区域
        x_vals = np.linspace(min(total_data), max(total_data), 100)
        kde = sns.kdeplot(total_data, color=color, linewidth=2)
        x_kde, y_kde = kde.get_lines()[-1].get_data()
        y_interp = np.interp(x_vals, x_kde, y_kde)
        plt.fill_between(x_vals, 0, y_interp, color=color, alpha=0.2)

    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel("AUC", fontsize=16, weight='bold')
    plt.ylabel("Density", fontsize=16, weight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(xmin, xmax)
    plt.ylim(0, None)
    plt.legend(title='Model Size', fontsize=12, title_fontsize='13', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def draw_by_feature(csv_results, dataset_name, plot_title):
    """
    绘制特定数据集，不同特征在同一图中的 AUC 值随长度变化的图
    并计算平均值和方差，用 fill_between 绘制方差，同时绘制单个结果
    """
    df_filtered = csv_results[csv_results["dataset"] == dataset_name]
    unique_features = df_filtered['feature'].unique()
    for feature in unique_features:
        df_feature = df_filtered[df_filtered['feature'] == feature]
        plt.figure(figsize=(12, 8))

        # 绘制单个结果
        sns.lineplot(data=df_feature, x="length", y="auc", hue="model_size", units="seed", estimator=None,
                     palette="viridis", linewidth=1, alpha=0.3, legend=False)

        # 按照 length 和 model_size 进行分组，计算平均值和标准差
        grouped = df_feature.groupby(['length', 'model_size'], as_index=False).agg({'auc': ['mean', 'std']})
        grouped.columns = ['length', 'model_size', 'mean_auc', 'std_auc']

        # 绘制平均曲线
        sns.lineplot(data=grouped, x="length", y="mean_auc", hue="model_size", marker="o", palette="viridis")

        # 使用 fill_between 绘制方差区域
        model_sizes = grouped['model_size'].unique()
        palette = dict(zip(model_sizes, sns.color_palette("viridis", n_colors=len(model_sizes))))

        for model_size in model_sizes:
            df_model = grouped[grouped['model_size'] == model_size]
            lengths = df_model['length']
            mean_auc = df_model['mean_auc']
            std_auc = df_model['std_auc']
            color = palette[model_size]
            plt.fill_between(lengths, mean_auc - std_auc, mean_auc + std_auc, color=color, alpha=0.2)

        plt.title(f"{plot_title} - {feature}", fontsize=16)
        plt.xlabel("Length", fontsize=14)
        plt.ylabel("AUC", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Model Size', fontsize=12, title_fontsize='13')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()


def plot_hist_by_category(df, category, title, bins=30, xmin=0.5, xmax=1.0, show_hist=False):
    """ 绘制指定分类的AUC概率密度函数图，汇总所有种子的数据 并绘制均值和方差区域 """
    plt.figure(figsize=(12, 8))
    sns.set_palette("Set2")
    unique_categories = df[category].unique()

    if len(unique_categories) > 10:
        sns.set_palette(sns.color_palette("husl", len(unique_categories)))

    colors = sns.color_palette()

    for i, cat in enumerate(unique_categories):
        subset = df[df[category] == cat]
        color = colors[i % len(colors)]

        data_by_seed = [subset[subset['seed'] == seed]['auc'] for seed in subset['seed'].unique()]

        # 将所有种子的 AUC 数据汇总
        all_data = np.concatenate(data_by_seed)

        # 计算均值和标准差
        mean_auc = all_data.mean()
        std_auc = all_data.std()

        # 计算密度估计
        x_vals = np.linspace(xmin, xmax, 500)
        all_interp_ys = np.array([np.interp(x_vals, *sns.kdeplot(data, color=color, alpha=0.1, linewidth=0).get_lines()[
            -1].get_data(), left=0, right=0) for data in data_by_seed])
        mean_y = all_interp_ys.mean(axis=0)
        std_y = all_interp_ys.std(axis=0)

        # 绘制总体的PDF均值
        sns.lineplot(x=x_vals, y=mean_y, color=color, label=f"{cat}", linewidth=2)

        # 使用填充区域展示均值 ± 方差
        plt.fill_between(x_vals, mean_y - std_y, mean_y + std_y, color=color, alpha=0.2)

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
    """ 对比不同数据集在相同特征上的概率密度函数 并绘制均值和方差区域 """
    plt.figure(figsize=(14, 10))
    sns.set_palette("Set2")
    dataset_labels = ["Truncated", "Relative", "Untruncated"]
    colors = sns.color_palette("Set2")

    for i, (df, label) in enumerate(zip(datasets, dataset_labels)):
        df_filtered = df[(df["dataset"] == dataset_name) & (df["feature"] == feature)]
        color = colors[i % len(colors)]

        data_by_seed = [df_filtered[df_filtered['seed'] == seed]['auc'] for seed in df_filtered['seed'].unique()]

        # 将所有种子的 AUC 数据汇总
        all_data = np.concatenate(data_by_seed)

        # 计算均值和标准差
        mean_auc = all_data.mean()
        std_auc = all_data.std()

        # 计算密度估计
        x_vals = np.linspace(xmin, xmax, 500)
        all_interp_ys = np.array([np.interp(x_vals, *sns.kdeplot(data, color=color, alpha=0.1, linewidth=0).get_lines()[
            -1].get_data(), left=0, right=0) for data in data_by_seed])
        mean_y = all_interp_ys.mean(axis=0)
        std_y = all_interp_ys.std(axis=0)

        # 绘制总体的PDF均值
        sns.lineplot(x=x_vals, y=mean_y, color=color, label=f"{label}", linewidth=2)

        # 使用填充区域展示均值 ± 方差
        plt.fill_between(x_vals, mean_y - std_y, mean_y + std_y, color=color, alpha=0.2)

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


def plot_pdf_comparison_by_specified_feature(datasets, features, title, bins=30, xmin=0.5, xmax=1.0, compare_target="model_size"):
    """ 对比不同方法在所有特征上的概率密度函数 并绘制均值和方差区域 """
    plt.figure(figsize=(14, 10))
    sns.set_palette("Set2")
    method_labels = ["Truncated", "Relative", "Untruncated"]
    colors = sns.color_palette("Set2", len(features))

    for j, model_size in enumerate(features):
        color = colors[j % len(colors)]

        all_kde_data = []

        # 对每个数据集（方法）进行操作
        for df in datasets:
            df_filtered = df[df[compare_target] == model_size]

            # 获取属于特定模型大小的所有种子
            seeds = df_filtered['seed'].unique()

            for seed in seeds:
                seed_filtered = df_filtered[df_filtered['seed'] == seed]
                data = seed_filtered['auc']

                # 绘制单个种子的PDF曲线，但设置透明度低一点
                kde = sns.kdeplot(data, color=color, alpha=0.1, linewidth=0.5)
                kde_data = kde.get_lines()[-1].get_data()
                all_kde_data.append(kde_data)

        # 准备x_vals
        x_vals = np.linspace(xmin, xmax, 500)

        # 将所有的kde_data进行插值到相同的x坐标上
        all_interp_ys = np.array([np.interp(x_vals, kde_x, kde_y, left=0, right=0) for kde_x, kde_y in all_kde_data])

        # 计算均值和标准差
        mean_y = all_interp_ys.mean(axis=0)
        std_y = all_interp_ys.std(axis=0)

        # 绘制总体的PDF均值
        sns.lineplot(x=x_vals, y=mean_y, color=color, label=f"{model_size}", linewidth=2)

        # 使用填充区域展示均值 ± 方差
        plt.fill_between(x_vals, mean_y - std_y, mean_y + std_y, color=color, alpha=0.2)

    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel("AUC", fontsize=16, weight='bold')
    plt.ylabel("Density", fontsize=16, weight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(xmin, xmax)
    plt.ylim(0, None)
    plt.legend(title='Model Size', fontsize=12, title_fontsize='13', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def plot_cdf(data, label, title, bins=30, xmin=0.5, xmax=1.0):
    """
    绘制累计分布函数（CDF）
    """
    plt.figure(figsize=(12, 8))
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    cdf = np.cumsum(hist * np.diff(bin_edges))
    sns.lineplot(x=bin_edges[1:], y=cdf, label=label)
    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel("AUC", fontsize=16, weight='bold')
    plt.ylabel("CDF", fontsize=16, weight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(xmin, xmax)
    plt.ylim(0, 1)
    plt.legend(title='Dataset Type', fontsize=12, title_fontsize='13', loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_cdfs(datasets, dataset_labels, title, bins=30, xmin=0.5, xmax=0.6):
    """
    绘制多个数据集的CDF
    """
    plt.figure(figsize=(14, 10))
    sns.set_palette("Set2")
    for i, (df, label) in enumerate(zip(datasets, dataset_labels)):
        data = df['auc']
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        cdf = np.cumsum(hist * np.diff(bin_edges))
        sns.lineplot(x=bin_edges[1:], y=cdf, label=label)
    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel("AUC", fontsize=16, weight='bold')
    plt.ylabel("CDF", fontsize=16, weight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(xmin, xmax)
    plt.ylim(0, 1)
    plt.legend(title='Method', fontsize=12, title_fontsize='13', loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# 准备绘图函数所需的数据集和标签
datasets = [truncated_results, relative_results, untruncated_results]
dataset_labels = ["Truncated", "Relative", "Untruncated"]
shared_datasets = list(set(relative_results['dataset'].unique()) & set(truncated_results['dataset'].unique()) & set(
    untruncated_results['dataset'].unique()))
shared_model_sizes = list(
    set(truncated_results['model_size'].unique()) & set(relative_results['model_size'].unique()) & set(
        untruncated_results['model_size'].unique()))

# ----------------------------------------------------------------------------------------------------------------------
# 示例绘图调用

# 绘制特定数据集的AUC值随长度变化的图，包含方差区域和单个结果
#draw_by_dataset_and_model_size(truncated_results, "Pile-CC", "Absolute AUC for Pile-CC by Model Size")
#draw_by_dataset_and_model_size(relative_results, "Pile-CC", "Relative AUC for Pile-CC by Model Size")
#draw_by_dataset_and_model_size(untruncated_results, "Pile-CC", "Untruncated AUC for Pile-CC by Model Size")

# 绘制特定数据集，不同特征的图
#draw_by_feature(truncated_results, "Github", "Absolute AUC for Github by Model Size and Feature")

# 绘制所有数据集的AUC概率密度函数，并展示均值和方差
plot_hist_by_category(relative_results, "dataset",
                      "Probability Density Function of AUC for All Datasets (Relative Results)", bins=30, xmin=0.5,
                      xmax=0.6, show_hist=False)
plot_hist_by_category(truncated_results, "dataset",
                      "Probability Density Function of AUC for All Datasets (Truncated Results)", bins=30, xmin=0.5,
                      xmax=0.6, show_hist=False)
plot_hist_by_category(untruncated_results, "dataset",
                      "Probability Density Function of AUC for All Datasets (Untruncated Results)", bins=30, xmin=0.5,
                      xmax=0.6, show_hist=False)

# 对比特定数据集和特征在不同方法下的概率密度函数，并展示均值和方差
plot_pdf_comparison(datasets, "Wikipedia (en)", "loss",
                    "Probability Density Function Comparison for Wikipedia (en) - Loss Feature", bins=30, xmin=0.5,
                    xmax=0.6, show_hist=False)

# 对比不同方法在所有特征上的概率密度函数，并展示均值和方差
plot_pdf_comparison_by_specified_feature(datasets, truncated_results["feature"].unique(),
                                "Probability Density Function Comparison of Different Methods", bins=30, xmin=0.5,
                                xmax=0.6, compare_target="feature")

plot_pdf_comparison_by_specified_feature(datasets, shared_datasets,
                                "Probability Density Function Comparison of Different Methods", bins=30, xmin=0.5,
                                xmax=0.6, compare_target="dataset")
plot_pdf_comparison_by_specified_feature(datasets, truncated_results["model_size"].unique(),
                                "Probability Density Function Comparison of Different Methods", bins=30, xmin=0.5,
                                xmax=0.6, compare_target="model_size")

# 绘制特定数据集的CDF
#plot_cdf(truncated_results[truncated_results["dataset"] == "Wikipedia (en)"]["auc"], "Truncated Wikipedia (en)",
#         "CDF of Wikipedia (en) AUC for Truncated")

# 对比不同方法的CDF
#plot_cdfs(datasets, dataset_labels, "CDF Comparison of Different Methods", bins=30, xmin=0.5, xmax=0.6)



