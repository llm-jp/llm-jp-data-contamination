from matplotlib import pyplot as plt
import pandas as pd


for model_size in ["160m", "410m", "1b", "2.8b", "6.9b", "12b"]:
    df = pd.read_csv(f"embedding_results_online/{model_size}_embedding_result.csv")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('DB Index')
    ax1.tick_params(axis='y')
    for idx, dataset_name in enumerate(["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc","pubmed_central", "wikipedia_(en)"]):
        db_index_value = df[df["Dataset Name"]==dataset_name]["DB Index"].tolist()
        ax1.plot(db_index_value, color=color_list[idx], label=dataset_name)
    fig.tight_layout()  # to prevent label overlap
    plt.title(f"Embedding Results {model_size}")
    plt.grid(True)
    plt.legend()
    plt.show()