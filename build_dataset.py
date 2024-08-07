from datasets import load_dataset
import os
import torch
from collections import defaultdict
from tqdm import tqdm
import time
import pdb

def process_and_save_dataset(ds, name, items_per_file=250000, batch_size=10000):
    file_counters = defaultdict(int)
    grouped_by_meta = defaultdict(list)
    count = 0

    while True:
        batch = list(ds.take(batch_size))
        if not batch:
            break
        start_time = time.time()
        for example in batch:
            meta_name = example['meta']["pile_set_name"]
            grouped_by_meta[meta_name].append(example["text"])

            # When batch size limit is reached
            if len(grouped_by_meta[meta_name]) >= items_per_file:
                filename = f"/model/pile/by_dataset/{name}_{meta_name}_{file_counters[meta_name]}.pt"

                # Check if the file already exists
                # if not os.path.exists(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                torch.save(grouped_by_meta[meta_name], filename)
                print("Saved", filename)
                #pdb.set_trace()

                # Reset current group
                grouped_by_meta[meta_name].clear()
                file_counters[meta_name] += 1

        end_time = time.time()  # 运行完毕后再次获取当前时间戳
        elapsed_time = end_time - start_time  # 计算两次时间戳之间的差值，即运行时间
        count += len(batch)
        print(f"Processed {count} examples with {elapsed_time:.2f} seconds")
        if len(batch) < batch_size:
            break
    # Save remaining data
    for key in grouped_by_meta.keys():
        if len(grouped_by_meta[key])>0:  # Save if not empty
            filename = f"/model/pile/by_dataset/{name}_{key}_{file_counters[key]}.pt"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            torch.save(grouped_by_meta[key], filename)
            print("Saved", filename)


ds_valid = load_dataset("monology/pile-uncopyrighted", cache_dir="/model/pile", split="validation", streaming=True)
ds_test = load_dataset("monology/pile-uncopyrighted", cache_dir="/model/pile", split="test", streaming=True)
#ds_train = load_dataset("monology/pile-uncopyrighted", cache_dir="/model/pile", split="train", streaming=True)

process_and_save_dataset(ds_valid, "valid", items_per_file=100000000, batch_size=10000)
process_and_save_dataset(ds_test, "test", items_per_file=100000000, batch_size=10000)
#process_and_save_dataset(ds_train, "train", items_per_file=100000, batch_size=10000)