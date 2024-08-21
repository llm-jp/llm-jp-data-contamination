import datasets
from utils import *
import argparse

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
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--dir", type=str, default="absolute_filtered_result")
parser.add_argument("--load_dir", type=str, default="absolute_filtered_dataset")
parser.add_argument("--generation_batch_size", type=int, default=1)
parser.add_argument("--truncated", type=str, default="truncated", choices=["truncated", "untruncated", "both"])
parser.add_argument("--generation_samples", type=int, default=10)
parser.add_argument("--max_input_tokens", type=int, default=512)
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--temperature", type=float, default=0.8)
args = parser.parse_args()

dataset_names = get_dataset_list(args.dataset_name)
for dataset_name in dataset_names:
    print(dataset_name)
    for min_len in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
        args.min_len = min_len
        dataset = obtain_dataset(dataset_name, args)
        print("average member length", sum([len(x.split()) for x in dataset["member"]])/len(dataset["member"]))
        print("average nonmember length", sum([len(x.split()) for x in dataset["nonmember"]])/len(dataset["nonmember"]))
        print("member set size:", len(dataset["member"]))
        print("nonmember set size:", len(dataset["nonmember"]))
        if len(dataset["member"]) < 300 or len(dataset["nonmember"]) < 300:
            print("too small")
