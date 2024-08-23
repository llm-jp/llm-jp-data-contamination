import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

dataset_names = [
    "arxiv", "dm_mathematics", "github", "WikiMIAall"
]
model_size_list = ["12b"]
continuation_size = 32

for context_size in [32]:
    fig, axs = plt.subplots(1, len(dataset_names), figsize=(12, 4), sharey=True)
    fig.suptitle(f"Memorization Score Distribution for Different Datasets)")

    for idx, dataset_name in enumerate(dataset_names):
        table = pd.read_csv(f"mem_score_online/{model_size_list[0]}/{dataset_name}_{context_size}_{continuation_size}_mem_score.csv", index_col=0)
        member_table = table[table["set_name"] == "member"]
        nonmember_table = table[table["set_name"] == "nonmember"]
        member_scores = member_table["mem_score"].values
        nonmember_scores = nonmember_table["mem_score"].values

        bins = np.arange(0, 1.1, 0.1)  # 包含1.0
        member_hist, _ = np.histogram(member_scores, bins=bins)
        nonmember_hist, _ = np.histogram(nonmember_scores, bins=bins)

        member_hist = member_hist / np.sum(member_hist)
        nonmember_hist = nonmember_hist / np.sum(nonmember_hist)

        bar_width = 0.04
        pos = bins[:-1]
        member_pos = pos - bar_width / 2
        nonmember_pos = pos + bar_width / 2

        axs[idx].bar(member_pos, member_hist, width=bar_width, alpha=0.6, label='member', edgecolor='black')
        axs[idx].bar(nonmember_pos, nonmember_hist, width=bar_width, alpha=0.6, label='nonmember', edgecolor='black')
        axs[idx].set_title(f"{dataset_name} ({model_size_list[0]})")
        axs[idx].set_xlabel('Score')
        axs[idx].set_ylabel('Ratio')
        axs[idx].set_xticks(bins)
        axs[idx].legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"mem_score_online/analysis.png", dpi=800)
    plt.show()