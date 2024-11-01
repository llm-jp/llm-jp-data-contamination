import argparse
from gray_box_mia import compute_gray_box_method
from black_box_mia import compute_black_box_mia
from eda_pac_mia import compute_eda_pac
from recall_mia import compute_recall
import random

random.seed(42)
parser = argparse.ArgumentParser()
##common arguments
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_length", type=int, default=2048)
parser.add_argument("--model_size", type=str, default="1b")
parser.add_argument("--dataset_name", type=str, default="Pile-CC", choices=["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                     "pubmed_central", "wikipedia_(en)", "full_pile", "all","ArXiv", "Enron Emails", "FreeLaw", 'Gutenberg (PG-19)', 'NIH ExPorter', "Pile-CC",'PubMed Central',
                'Ubuntu IRC', 'Wikipedia (en)', 'DM Mathematics', "EuroParl", "Github","HackerNews", "PhilPapers",
                "PubMed Abstracts", "StackExchange", "local_all"])
parser.add_argument("--cuda", type=int, default=0, help="cuda device")
parser.add_argument("--min_len", type=int, default=100)
parser.add_argument("--local_data", type=bool, default=True)
parser.add_argument("--relative", type=str, default="absolute", choices=["absolute", "relative"])
parser.add_argument("--truncated", type=str, default="truncated", choices=["truncated", "untruncated"])
parser.add_argument("--same_length", action='store_false')
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--dataset_idx", type=int, default=1)
parser.add_argument("--gray", type=str, default="gray", choices=["gray", "black", "pac", "recall"])
parser.add_argument("--save_dir", type=str, default="mia_dataset_results")
parser.add_argument("--load_dir", type=str, default="mia_dataset")
#gray box mia
parser.add_argument("--refer_model", type=str, default="True")
parser.add_argument("--refer_cuda", type=int, default=7, help="cuda device")
#recall mia
parser.add_argument("--num_shots", type=int, default=12)
parser.add_argument("--pass_window", type=bool, default=False)
#black box mia
parser.add_argument("--generation_batch_size", type=int, default=1)
parser.add_argument("--generation_samples", type=int, default=10)
parser.add_argument("--max_input_tokens", type=int, default=512)
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--temperature", type=float, default=0.8)
args = parser.parse_args()
print(args)
if args.gray == "gray":
    compute_gray_box_method(args)
elif args.gray == "black":
    compute_black_box_mia(args)
elif args.gray == "pac":
    compute_eda_pac(args)
elif args.gray == "recall":
    compute_recall(args)

