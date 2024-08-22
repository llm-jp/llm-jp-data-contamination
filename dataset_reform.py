import datasets
from utils import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_length", type=int, default=2048)
parser.add_argument("--model_size", type=str, default="160m")
parser.add_argument("--dataset_name", type=str, default="local_all", choices=["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                     "pubmed_central", "wikipedia_(en)", "full_pile", "all","ArXiv", "Enron Emails", "FreeLaw", 'Gutenberg (PG-19)', 'NIH ExPorter', "Pile-CC",'PubMed Central',
                'Ubuntu IRC', 'Wikipedia (en)', 'DM Mathematics', "EuroParl", "Github","HackerNews", "PhilPapers",
                "PubMed Abstracts", "StackExchange", "local_all"])
parser.add_argument("--cuda", type=int, default=0, help="cuda device")
parser.add_argument("--refer_cuda", type=int, default=7, help="cuda device")
parser.add_argument("--min_len", type=int, default=100)
parser.add_argument("--local_data", type=bool, default=True)
parser.add_argument("--same_length", action='store_false')
parser.add_argument("--relative", type=str, default="absolute", choices=["absolute", "relative"])
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--dir", type=str, default="relative_filtered_result")
parser.add_argument("--load_dir", type=str, default="relative_filtered_dataset")
parser.add_argument("--generation_batch_size", type=int, default=1)
parser.add_argument("--truncated", type=str, default="nontruncated", choices=["truncated", "untruncated", "both"])
args = parser.parse_args()

dataset_names = get_dataset_list(args)
for dataset_name in dataset_names:
    args.dataset_name = dataset_name
    print(dataset_name)
    dataset_indicator = True
    if "absolute" in args.dir:
        enumerate_list = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    else:
        enumerate_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    for min_len in enumerate_list:
        args.min_len = min_len
        dataset = obtain_dataset(dataset_name, args)
        if len(dataset["member"]) == 0 or len(dataset["nonmember"]) == 0:
            print("empty")
            continue
        print("average member length", sum([len(x.split()) for x in dataset["member"]])/len(dataset["member"]))
        print("average nonmember length", sum([len(x.split()) for x in dataset["nonmember"]])/len(dataset["nonmember"]))
        print("member set size:", len(dataset["member"]))
        print("nonmember set size:", len(dataset["nonmember"]))
        if len(dataset["member"]) < 100 or len(dataset["nonmember"]) < 100:
            print("too small")
            dataset_indicator = False
    if dataset_indicator:
        print("dataset is good")
        added_address = args.truncated
        os.makedirs(f"./mia_dataset_{added_address}/{dataset_name}", exist_ok=True)
        for min_len in enumerate_list:
            if args.relative == "relative":
                dataset = datasets.load_from_disk(f"./{args.load_dir}/{min_len}/{dataset_name}")
                dataset.save_to_disk(f"./mia_dataset_relative/{dataset_name}/{min_len}")
            else:
                min_len = min_len if min_len != 0 else 5
                max_len = 100 if min_len == 5 else min_len + 100

                dataset = datasets.load_from_disk(f"./{args.load_dir}/{min_len}_{max_len}_{added_address}/{dataset_name}")
                dataset.save_to_disk(f"./mia_dataset_{added_address}/{dataset_name}/{min_len}_{max_len}")
