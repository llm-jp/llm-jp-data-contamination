import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

dataset_names = [
    "arxiv", "pile_cc", "pubmed_central",
    "dm_mathematics", "github", "WikiMIAall",
]
model_size_list = ["12b"]
continuation_size = 32
context_size = 32

fig, axs = plt.subplots(2, len(dataset_names)//2, figsize=(12, 6), sharey=True)
fig.suptitle("Memorization Score Distribution for Different Datasets")

# Split rows for indifferentiable and differentiable domains
indiff_dom = dataset_names[:len(dataset_names)//2]
diff_dom = dataset_names[len(dataset_names)//2:]

# Plotting indifferentiable domain
for idx, dataset_name in enumerate(indiff_dom):
    table = pd.read_csv(f"mem_score_online/{model_size_list[0]}/{dataset_name}_{context_size}_{continuation_size}_mem_score.csv", index_col=0)
    member_table = table[table["set_name"] == "member"]
    nonmember_table = table[table["set_name"] == "nonmember"]
    member_scores = member_table["mem_score"].values
    nonmember_scores = nonmember_table["mem_score"].values
    bins = np.arange(0, 1.1, 0.1)
    member_hist, _ = np.histogram(member_scores, bins=bins)
    nonmember_hist, _ = np.histogram(nonmember_scores, bins=bins)
    member_hist = member_hist / np.sum(member_hist)
    nonmember_hist = nonmember_hist / np.sum(nonmember_hist)
    bar_width = 0.04
    pos = bins[:-1]
    member_pos = pos - bar_width / 2
    nonmember_pos = pos + bar_width / 2
    axs[0, idx].bar(member_pos, member_hist, width=bar_width, alpha=0.6, label='member', edgecolor='black')
    axs[0, idx].bar(nonmember_pos, nonmember_hist, width=bar_width, alpha=0.6, label='nonmember', edgecolor='black')
    axs[0, idx].set_title(f"{dataset_name} ({model_size_list[0]})")
    axs[0, idx].set_xlabel('Score')
    axs[0, idx].set_ylabel('Ratio')
    axs[0, idx].set_xticks(bins)
    axs[0, idx].legend(loc='upper right')
axs[0, 0].annotate("Indifferentiable Domain", xy=(0, 0.5), xytext=(-axs[0, 0].yaxis.labelpad - 5, 0),
    xycoords=axs[0, 0].yaxis.label, textcoords='offset points',
    size='large', ha='center', va='center', rotation=90)

# Plotting differentiable domain
for idx, dataset_name in enumerate(diff_dom):
    table = pd.read_csv(f"mem_score_online/{model_size_list[0]}/{dataset_name}_{context_size}_{continuation_size}_mem_score.csv", index_col=0)
    member_table = table[table["set_name"] == "member"]
    nonmember_table = table[table["set_name"] == "nonmember"]
    member_scores = member_table["mem_score"].values
    nonmember_scores = nonmember_table["mem_score"].values
    bins = np.arange(0, 1.1, 0.1)
    member_hist, _ = np.histogram(member_scores, bins=bins)
    nonmember_hist, _ = np.histogram(nonmember_scores, bins=bins)
    member_hist = member_hist / np.sum(member_hist)
    nonmember_hist = nonmember_hist / np.sum(nonmember_hist)
    bar_width = 0.04
    pos = bins[:-1]
    member_pos = pos - bar_width / 2
    nonmember_pos = pos + bar_width / 2
    axs[1, idx].bar(member_pos, member_hist, width=bar_width, alpha=0.6, label='member', edgecolor='black')
    axs[1, idx].bar(nonmember_pos, nonmember_hist, width=bar_width, alpha=0.6, label='nonmember', edgecolor='black')
    axs[1, idx].set_title(f"{dataset_name} ({model_size_list[0]})")
    axs[1, idx].set_xlabel('Score')
    axs[1, idx].set_ylabel('Ratio')
    axs[1, idx].set_xticks(bins)
    axs[1, idx].legend(loc='upper right')
axs[1, 0].annotate("Differentiable Domain", xy=(0, 0.5), xytext=(-axs[1, 0].yaxis.labelpad - 5, 0),
    xycoords=axs[1, 0].yaxis.label, textcoords='offset points',
    size='large', ha='center', va='center', rotation=90)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("mem_score_online/analysis.png", dpi=800)
plt.show()