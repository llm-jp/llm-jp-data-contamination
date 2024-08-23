import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

dataset_names = ["wikipedia_(en)", "pile_cc", "arxiv", "dm_mathematics", "github", "hackernews", "pubmed_central", "full_pile", "WikiMIA64", "WikiMIA128", "WikiMIA256", "WikiMIAall"]
#model_size_list = ["160m", "410m", "1b", "2.8b", "6.9b", "12b"]
model_size_list = ["12b"]
continuation_size = 32

for context_size in [8, 16, 32, 48, 64]:
    for dataset_name in dataset_names:
        fig, axs = plt.subplots(1, len(model_size_list), figsize=(24, 6), sharey=True)
        fig.suptitle(f"{dataset_name} Mem Score Distribution (Context Size: {context_size})")

        for i, model_size in enumerate(model_size_list):
            table = pd.read_csv(f"mem_score_online/{model_size}/{dataset_name}_{context_size}_{continuation_size}_mem_score.csv", index_col=0)
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

            axs[i].bar(member_pos, member_hist, width=bar_width, alpha=0.6, label='member', edgecolor='black')
            axs[i].bar(nonmember_pos, nonmember_hist, width=bar_width, alpha=0.6, label='nonmember', edgecolor='black')
            axs[i].set_title(f"{model_size} Model")

            axs[i].set_xlabel('Score')
            if i == 0:
                axs[i].set_ylabel('Frequency')
            axs[i].set_xticks(bins)
            axs[i].legend(loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #plt.savefig(f"mem_score_online/{dataset_name}_{context_size}_{continuation_size}_mem_score.png")
        plt.show()