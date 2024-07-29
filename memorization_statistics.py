import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


dataset_names = ["wikipedia_(en)", "pile_cc", "arxiv", "dm_mathematics", "github", "hackernews", "pubmed_central", "full_pile"]
model_size = "410m"
for dataset_name in dataset_names:
    table = pd.read_csv(f"mem_score_online/{model_size}/{dataset_name}_mem_score.csv", index_col=0)
    member_table= table[table["set_name"] == "member"]
    nonmember_table = table[table["set_name"] == "nonmember"]
    member_scores = member_table["mem_score"].values
    nonmember_scores = nonmember_table["mem_score"].values
    bins = np.arange(0, 1.1, 0.1)  # 包含1.0

    plt.figure()
    plt.hist(member_scores, bins=bins, alpha=0.5, label='member', edgecolor='black')
    plt.hist(nonmember_scores, bins=bins, alpha=0.5, label='nonmember', edgecolor='black')
    plt.legend(loc='upper right')
    plt.title(f"{dataset_name} Mem Score Distribution")
    plt.xticks(bins)  # 设置X轴的刻度
    plt.savefig(f"mem_score_online/{model_size}/{dataset_name}_mem_score.png")
    plt.show()
