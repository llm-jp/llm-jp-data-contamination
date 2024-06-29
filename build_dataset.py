from datasets import load_dataset
import os
import json
from collections import defaultdict
import torch
from tqdm import tqdm

def process_and_save_dataset(ds, name):
    grouped_by_meta = defaultdict(list)

    # Group data by 'meta' attribute
    for example in tqdm(ds):
        grouped_by_meta[example['meta']["pile_set_name"]].append(example["text"])

    # Save each group to a separate file as PyTorch tensors
    for meta, dataset in grouped_by_meta.items():
        filename = f"/model/pile/by_dataset/{name}_{meta}.pt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(dataset, filename)

ds_train = load_dataset("monology/pile-uncopyrighted", cache_dir="/model/pile", split="train", streaming=True)
ds_valid = load_dataset("monology/pile-uncopyrighted", cache_dir="/model/pile", split="validation", streaming=True)
ds_test = load_dataset("monology/pile-uncopyrighted", cache_dir="/model/pile", split="test", streaming=True)

process_and_save_dataset(ds_train, "train")
process_and_save_dataset(ds_valid, "valid")
process_and_save_dataset(ds_test, "test")