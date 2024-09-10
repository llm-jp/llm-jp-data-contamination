import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Load the data
truncated_results = pd.read_csv("auc_results_absolute_truncated_all_length_all_model_size.csv")
relative_results = pd.read_csv("auc_results_relative_truncated_all_length_all_model_size.csv")
untruncated_results = pd.read_csv("auc_results_absolute_untruncated_all_length_all_model_size.csv")

# Set a Seaborn style for the plot
sns.set(style="whitegrid")


def draw_by_dataset(csv_results, dataset_name, plot_title):
    """
    Draw graph for a specific dataset from the csv_results.
    """
    df_filtered = csv_results[csv_results["dataset"] == dataset_name]

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x="length", y="auc", marker="o")

    # Customize the plot
    plt.title(plot_title, fontsize=16)
    plt.xlabel("Length", fontsize=14)
    plt.ylabel("AUC", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.show()


# Draw for the specific dataset "ArXiv" with an appropriate title
draw_by_dataset(truncated_results, "ArXiv", "Absolute AUC for ArXiv")


def draw_by_dataset_and_model_size(csv_results, dataset_name, plot_title):
    """
    Draw graph for specific dataset and include all model sizes in one graph.
    """
    df_filtered = csv_results[csv_results["dataset"] == dataset_name]

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Use seaborn's lineplot with hue to differentiate model sizes.
    sns.lineplot(data=df_filtered, x="length", y="auc", hue="model_size",
                 marker="o", palette="viridis")

    # Customize the plot
    plt.title(plot_title, fontsize=16)
    plt.xlabel("Length", fontsize=14)
    plt.ylabel("AUC", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Model Size', fontsize=12, title_fontsize='13')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.show()


# Draw for the specific dataset "ArXiv" with an appropriate title and all model sizes in one plot
draw_by_dataset_and_model_size(truncated_results, "Pile-CC", "Absolute AUC for ArXiv by Model Size")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
truncated_results = pd.read_csv("auc_results_absolute_truncated_all_length_all_model_size.csv")
relative_results = pd.read_csv("auc_results_relative_truncated_all_length_all_model_size.csv")
untruncated_results = pd.read_csv("auc_results_absolute_untruncated_all_length_all_model_size.csv")

# Set a Seaborn style for the plot
sns.set(style="whitegrid")


def draw_by_feature(csv_results, dataset_name, plot_title):
    """
    Draw graph grouped by feature and include all model sizes in one graph.
    """
    df_filtered = csv_results[csv_results["dataset"] == dataset_name]

    # Extract unique features
    unique_features = df_filtered['feature'].unique()

    for feature in unique_features:
        df_feature = df_filtered[df_filtered['feature'] == feature]

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Use seaborn's lineplot with hue to differentiate model sizes.
        sns.lineplot(data=df_feature, x="length", y="auc", hue="model_size",
                     marker="o", palette="viridis")

        # Customize the plot
        plt.title(f"{plot_title} - {feature}", fontsize=16)
        plt.xlabel("Length", fontsize=14)
        plt.ylabel("AUC", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Model Size', fontsize=12, title_fontsize='13')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.show()


# Draw for the specific dataset "ArXiv" with an appropriate title and by feature
draw_by_feature(truncated_results, "Github", "Absolute AUC for ArXiv by Model Size and Feature")

sns.set(style="whitegrid")


# Function to plot PDF for a specific category

def plot_hist_by_category(df, category, title, bins=30, xmin=0.5, xmax=1.0, show_hist=False):
    plt.figure(figsize=(12, 8))

    # 设置Seaborn样式
    sns.set_style("whitegrid")  # 白色网格背景
    sns.set_palette("Set2")  # 设置颜色调色板

    unique_categories = df[category].unique()
    # 如果类别数量较多，使用不同的调色板和样式
    if len(unique_categories) > 10:
        sns.set_palette(sns.color_palette("husl", len(unique_categories)))

    for i, cat in enumerate(unique_categories):
        subset = df[df[category] == cat]
        data = subset['auc']

        if show_hist:
            sns.histplot(data, bins=bins, kde=True, label=cat, stat='density', element='step', linewidth=1.5)
        else:
            sns.kdeplot(data, label=cat, linewidth=2, linestyle=('-' if i % 2 == 0 else '--'), alpha=0.7)

    # 设置标题和标签
    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel("AUC", fontsize=16, weight='bold')
    plt.ylabel("Density", fontsize=16, weight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # 设置x轴和y轴范围
    plt.xlim(xmin, xmax)
    plt.ylim(0, None)

    # 美化图例，将其放在图表外部
    plt.legend(title=category, fontsize=12, title_fontsize='13', loc='center left', bbox_to_anchor=(1, 0.5))

    # 显示网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 显示绘制的图形
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # 确保子图可以完整展示
    plt.show()


# Example for all datasets using histogram with KDE
plot_hist_by_category(relative_results, "dataset", "Probability Density Function of AUC for All Datasets", bins=30,
                      xmin=0.5, xmax=0.6, show_hist=False)
plot_hist_by_category(relative_results, "model_size", "Probability Density Function of AUC for All Model Sizes",
                      bins=30, xmin=0.5, xmax=0.6, show_hist=False)
plot_hist_by_category(relative_results, "feature", "Probability Density Function of AUC for All Features", bins=30,
                      xmin=0.5, xmax=0.6, show_hist=False)
plot_hist_by_category(relative_results, "length", "Probability Density Function of AUC for All Length", bins=30,
                      xmin=0.5, xmax=0.6, show_hist=False)