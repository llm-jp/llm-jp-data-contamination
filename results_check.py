import os
save_dir = "mia_dataset_results"
dataset_idx_list = [0 ,1, 2]
dataset_name = "local_all"
relative = "absolute"
truncated = "truncated"
model_size_list = ["160m", "410m", "1b", "2.8b", "6.9b", "12b"]
#["160m", "410m", "1b", "2.8b", "6.9b", "12b"]
if dataset_name == "local_all" and relative == "absolute":
    if truncated == "truncated":
        dataset_list = ['Wikipedia (en)', "StackExchange", 'PubMed Central', "Pile-CC", "HackerNews",
                    "Github", "FreeLaw", "EuroParl", 'DM Mathematics', "ArXiv", ]
    elif truncated == "untruncated":
    # dataset_list = ['Wikipedia (en)', "USPTO Backgrounds", "StackExchange", "Pile-CC", "Github", "FreeLaw"]
        dataset_list = ['Wikipedia (en)', "StackExchange", "Pile-CC", "Github", "FreeLaw"]
elif dataset_name == "local_all" and relative == "relative":
    dataset_list = ["Wikipedia (en)", "StackExchange", 'PubMed Central', "Pile-CC", "NIH ExPorter", "HackerNews",
                "Github", "FreeLaw", "Enron Emails", "DM Mathematics", "ArXiv"]
min_len_list = range(0, 1000, 100) if relative == "absolute" else range(0, 100, 10)
for model_size in model_size_list:
    for dataset_name in dataset_list:
        for dataset_idx in [0, 1, 2]:
            for min_len in min_len_list:
                for detection_method in ["loss", "ccd", "eda_pac", "recall"]:
                    if os.path.exists(
                            f"{save_dir}_{dataset_idx}/{dataset_name}/{relative}/{truncated}/{min_len}_{model_size}_{detection_method}_dict.pkl"):
                        pass
                    else:
                        print(f"{model_size} {dataset_name} {dataset_idx} {min_len} {detection_method} not finished")

