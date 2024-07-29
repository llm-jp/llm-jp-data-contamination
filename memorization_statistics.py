import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


dataset_names = ["wikipedia_(en)", "pile_cc", "arxiv", "dm_mathematics", "github", "hackernews", "pubmed_central",
                 "full_pile", "WikiMIA64", "WikiMIA128","WikiMIA256", "WikiMIAall"]
model_size_list = ["410m", "6.9b", "12b"]
for dataset_name in dataset_names:
    for model_size in model_size_list:
        table = pd.read_csv(f"mem_score_online/{model_size}/{dataset_name}_mem_score.csv", index_col=0)
        if "WikiMIA" in dataset_name:
            member_table = table[table["set_name"] == "train"]
            nonmember_table = table[table["set_name"] == "test"]
        else:
            member_table= table[table["set_name"] == "member"]
            nonmember_table = table[table["set_name"] == "nonmember"]
        member_scores = member_table["mem_score"].values
        nonmember_scores = nonmember_table["mem_score"].values
        bins = np.arange(0, 1.1, 0.1)  # 包含1.0

        # 计算每个区间的成员和非成员的计数
        member_hist, _ = np.histogram(member_table["mem_score"], bins=bins)
        nonmember_hist, _ = np.histogram(nonmember_table["mem_score"], bins=bins)

        # 生成每个条形的位置
        bar_width = 0.04
        pos = bins[:-1]  # Remove the last bin edge
        member_pos = pos - bar_width / 2
        nonmember_pos = pos + bar_width / 2

        # Draw the bars
        plt.figure(figsize=(10, 5))
        plt.bar(member_pos, member_hist, width=bar_width, alpha=0.5, label='member', edgecolor='black')
        plt.bar(nonmember_pos, nonmember_hist, width=bar_width, alpha=0.5, label='nonmember', edgecolor='black')

        plt.title(f"{dataset_name} Mem Score Distribution")
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.xticks(bins)
        plt.legend(loc='upper right')

        plt.savefig(f"mem_score_online/{model_size}/{dataset_name}_mem_score.png")
        plt.show()
