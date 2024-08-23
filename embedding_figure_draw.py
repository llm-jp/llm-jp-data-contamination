import matplotlib.pyplot as plt
import pandas as pd

# 手动选择深一点的颜色，确保线条清晰可见
db_index_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']  # 深蓝色调
accuracy_colors = ['#d62728', '#9467bd', '#8c564b']  # 深红色调

datasets_1 = ["github", "dm_mathematics", "WikiMIAall"]
datasets_2 = ["arxiv", "pile_cc", "pubmed_central"]

line_styles = ['-', '--', '-.', ':']  # 四种不同线型
markers = ['o', 's', 'D', '^']  # 四种不同标记

# 创建图表
fig, axes = plt.subplots(3, 2, figsize=(18, 12), dpi=300, sharey=True, constrained_layout=True)

for row, model_size in enumerate(["410m", "2.8b", "12b"]):
    # 读取数据
    df = pd.read_csv(f"embedding_results_online/{model_size}_embedding_result.csv")
    accuracy_results = pd.read_csv(f"embedding_learning/{model_size}/learning_results.csv")

    ax1 = axes[row, 0]
    ax3 = axes[row, 1]

    # 子图1 - 第一个数据集
    for idx, dataset_name in enumerate(datasets_1):
        db_index_value = df[df["Dataset Name"] == dataset_name]["DB Index"].tolist()
        ax1.plot(db_index_value, color=db_index_colors[idx], label=f'{dataset_name} DB Index', linestyle=line_styles[idx], linewidth=2, marker=markers[idx], markersize=6)

    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('DB Index', fontsize=12, color='tab:blue')
    ax1.set_ylim(0, 50)
    ax1.set_title(f"Differentiable Domains - Model Size: {model_size}", fontsize=14)
    ax1.grid(True)

    # 创建双Y轴，用于显示Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylim(0.4, 1)
    ax2.grid(True)
    ax2.set_ylabel('Accuracy', fontsize=12, color='tab:red')

    for idx, dataset_name in enumerate(datasets_1):
        dataset_accuracy = accuracy_results[accuracy_results["Dataset"] == dataset_name]
        ax2.plot(dataset_accuracy['Test Accuracy'].tolist(), label=f'{dataset_name} Test Accuracy', linestyle=line_styles[idx], color=accuracy_colors[idx], marker=markers[idx], linewidth=2, alpha=0.8)

    # 子图2 - 第二个数据集
    for idx, dataset_name in enumerate(datasets_2):
        db_index_value = df[df["Dataset Name"] == dataset_name]["DB Index"].tolist()
        ax3.plot(db_index_value, color=db_index_colors[idx], label=f'{dataset_name} DB Index', linestyle=line_styles[idx], linewidth=2, marker=markers[idx], markersize=6)

    ax3.set_xlabel('Layer Index', fontsize=12)
    ax3.set_ylabel('DB Index', fontsize=12, color='tab:blue')
    ax3.set_ylim(0, 50)
    ax3.set_title(f"Non-Differentiable Domains - Model Size: {model_size}", fontsize=14)
    ax3.grid(True)

    # 创建双Y轴，用于显示Accuracy
    ax4 = ax3.twinx()
    ax4.set_ylabel('Accuracy', fontsize=12, color='tab:red')
    ax4.set_ylim(0.4, 1)
    ax4.grid(True)

    for idx, dataset_name in enumerate(datasets_2):
        dataset_accuracy = accuracy_results[accuracy_results["Dataset"] == dataset_name]
        ax4.plot(dataset_accuracy['Test Accuracy'].tolist(), label=f'{dataset_name} Test Accuracy', linestyle=line_styles[idx], color=accuracy_colors[idx], marker=markers[idx], linewidth=2, alpha=0.8)

    # 合并第一个子图的图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=10, loc='upper left')

    # 合并第二个子图的图例
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_4, labels_3 + labels_4, fontsize=10, loc='upper left')

# 保存到文件
plt.savefig("embedding_results_online/all_models_embedding_result.png", bbox_inches='tight', dpi=300)