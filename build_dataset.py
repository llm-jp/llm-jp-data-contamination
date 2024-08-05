from datasets import load_dataset
import os
import torch
from collections import defaultdict
from tqdm import tqdm
import itertools
import multiprocessing as mp


def process_and_save_chunk(chunk, name, file_counters, items_per_file):
    grouped_by_meta = defaultdict(list)
    for example in chunk:
        meta_name = example['meta']["pile_set_name"]
        grouped_by_meta[meta_name].append(example["text"])

        if len(grouped_by_meta[meta_name]) >= items_per_file:
            filename = f"/model/pile/by_dataset/{name}_{meta_name}_{file_counters[meta_name]}.pt"

            # Check if the file already exists
            if not os.path.exists(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                torch.save(grouped_by_meta[meta_name], filename)

            grouped_by_meta[meta_name].clear()
            file_counters[meta_name] += 1

    # Save any remaining data
    for meta, dataset in grouped_by_meta.items():
        if dataset:  # Save if not empty
            filename = f"/model/pile/by_dataset/{name}_{meta}_{file_counters[meta]}.pt"
            if not os.path.exists(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                torch.save(dataset, filename)


def process_and_save_dataset(ds, name, items_per_file=250000, batch_size=10000, num_workers=4):
    file_counters = defaultdict(int)

    chunk = []
    pool = mp.Pool(num_workers)

    for example in tqdm(itertools.islice(ds, 0, None)):
        chunk.append(example)

        if len(chunk) >= batch_size:
            pool.apply_async(process_and_save_chunk, (chunk, name, file_counters, items_per_file))
            chunk = []

    if chunk:
        pool.apply_async(process_and_save_chunk, (chunk, name, file_counters, items_per_file))

    pool.close()
    pool.join()


ds_valid = load_dataset("monology/pile-uncopyrighted", cache_dir="/model/pile", split="validation", streaming=True)
ds_test = load_dataset("monology/pile-uncopyrighted", cache_dir="/model/pile", split="test", streaming=True)
ds_train = load_dataset("monology/pile-uncopyrighted", cache_dir="/model/pile", split="train", streaming=True)

process_and_save_dataset(ds_valid, "valid", items_per_file=100000000, batch_size=10000, num_workers=4)
process_and_save_dataset(ds_test, "test", items_per_file=100000000, batch_size=10000, num_workers=4)
process_and_save_dataset(ds_train, "train", items_per_file=250000, batch_size=10000, num_workers=4)