from datasets import load_dataset
import os
import json
from collections import defaultdict
import torch
from tqdm import tqdm

def process_and_save_dataset(ds, name, items_per_file=250000):
    grouped_by_meta = defaultdict(list)
    file_counters = defaultdict(int)

    for idx, example in tqdm(enumerate(ds)):
        grouped_by_meta[example['meta']["pile_set_name"]].append(example["text"])
        if len(grouped_by_meta[example['meta']["pile_set_name"]]) >= items_per_file:
            # Save current group to a separate file as PyTorch tensor
            temp = example['meta']["pile_set_name"]
            filename = f"/model/pile/by_dataset/{name}_{temp}_{file_counters[temp]}.pt"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            torch.save(grouped_by_meta[example['meta']["pile_set_name"]], filename)
            # Reset current group
            grouped_by_meta[example['meta']["pile_set_name"]].clear()
            file_counters[example['meta']["pile_set_name"]] += 1
    # Save each group to a separate file as PyTorch tensors
    for meta, dataset in grouped_by_meta.items():
        if dataset:  # save if not empty
            filename = f"/model/pile/by_dataset/{name}_{meta}_{file_counters[meta]}.pt"
            torch.save(dataset, filename)


#ds_valid = load_dataset("monology/pile-uncopyrighted", cache_dir="/model/pile", split="validation", streaming=True)
#ds_test = load_dataset("monology/pile-uncopyrighted", cache_dir="/model/pile", split="test", streaming=True)
ds_train = load_dataset("monology/pile-uncopyrighted", cache_dir="/model/pile", split="train", streaming=True)

#process_and_save_dataset(ds_valid, "valid", items_per_file=1000000000)
#process_and_save_dataset(ds_test, "test", items_per_file=1000000000)
process_and_save_dataset(ds_train, "train")
