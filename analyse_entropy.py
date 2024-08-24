import numpy as np
import matplotlib.pyplot as plt

dataset_names = [
    "arxiv", "pile_cc", "dm_mathematics", "WikiMIAall"
]
model_size_list = ["12b", "6.9b"]

# 计算所有数据集和模型熵的最大最小值
min_entropy, max_entropy = float('inf'), float('-inf')
min_accumulated_diff, max_accumulated_diff = float('inf'), float('-inf')

for dataset_name in dataset_names:
    for model_size in model_size_list:
        member_entropy = np.load(f"entropy_online/{model_size}/{dataset_name}_member_entropy.npy")
        nonmember_entropy = np.load(f"entropy_online/{model_size}/{dataset_name}_non_member_entropy.npy")
        member_entropy_mean = np.mean(member_entropy, axis=0)
        nonmember_entropy_mean = np.mean(nonmember_entropy, axis=0)
        min_entropy = min(min_entropy, member_entropy_mean.min(), nonmember_entropy_mean.min())
        max_entropy = max(max_entropy, member_entropy_mean.max(), nonmember_entropy_mean.max())
        accumulated_diff = np.abs(np.cumsum(member_entropy_mean - nonmember_entropy_mean))
        min_accumulated_diff = min(min_accumulated_diff, accumulated_diff.min())
        max_accumulated_diff = max(max_accumulated_diff, accumulated_diff.max())

# 添加适当的padding
padding_entropy = 0.05 * (max_entropy - min_entropy)
min_entropy -= padding_entropy
max_entropy += padding_entropy
padding_accumulated_diff = 0.05 * (max_accumulated_diff - min_accumulated_diff)
min_accumulated_diff -= padding_accumulated_diff
max_accumulated_diff += padding_accumulated_diff


def plot_datasets(datasets, title):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴

    for dataset_name in datasets:
        fill_between_legend_added = False  # 每个数据集单独处理fill_between图例
        for idx, model_size in enumerate(model_size_list):
            member_entropy = np.load(f"entropy_online/{model_size}/{dataset_name}_member_entropy.npy")
            nonmember_entropy = np.load(f"entropy_online/{model_size}/{dataset_name}_non_member_entropy.npy")
            # 计算平均值
            member_entropy_mean = np.mean(member_entropy, axis=0)
            nonmember_entropy_mean = np.mean(nonmember_entropy, axis=0)
            print(f"Mean of Member Entropy for {model_size} - {dataset_name}: {member_entropy_mean}")
            print(f"Mean of Non-Member Entropy for {model_size} - {dataset_name}: {nonmember_entropy_mean}")
            # 绘制成员和非成员熵的均值曲线
            ax1.plot(range(len(member_entropy_mean)), member_entropy_mean,
                     label=f'{model_size} {dataset_name} Member Entropy')
            ax1.plot(range(len(nonmember_entropy_mean)), nonmember_entropy_mean,
                     label=f'{model_size} {dataset_name} Non-Member Entropy')
            # 填充成员和非成员熵之间的区域
            if idx == 0:
                ax1.fill_between(range(len(member_entropy_mean)), member_entropy_mean, nonmember_entropy_mean,
                                 alpha=0.2, label=f'{dataset_name} Difference')
            else:
                ax1.fill_between(range(len(member_entropy_mean)), member_entropy_mean, nonmember_entropy_mean,
                                 alpha=0.2)
            # 计算累积差异并绘制其曲线
            accumulated_diff = np.abs(np.cumsum(member_entropy_mean - nonmember_entropy_mean))
            ax2.plot(range(len(accumulated_diff)), accumulated_diff, linestyle='--',
                     label=f'{model_size} {dataset_name} Accumulated Difference')

        # 提取累积差异的y数据

    # 设置y轴取值范围
    ax1.set_ylim(min_entropy, max_entropy)
    ax2.set_ylim(min_accumulated_diff, max_accumulated_diff)
    # 设置主y轴的标签和标题
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Mean Entropy')
    ax1.set_title(title)
    ax1.grid(True)
    # 设置第二个y轴的标签
    ax2.set_ylabel('Accumulated Difference')
    # 合并两个y轴的图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.show()


# Plot the first two datasets
plot_datasets(dataset_names[:2], 'Mean Entropy of Member and Non-Member for First Two Datasets')

# Plot the next datasets
plot_datasets(dataset_names[2:], 'Mean Entropy of Member and Non-Member for Next Two Datasets')