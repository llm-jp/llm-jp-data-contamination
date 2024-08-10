import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--model_size", type=str, default="410m")
parser.add_argument("--dataset_name", type=str, default="ArXiv", choices=["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                     "pubmed_central", "wikipedia_(en)", "full_pile", "all"])
parser.add_argument("--cuda", type=int, default=1, help="cuda device")
parser.add_argument("--skip_calculation", type=str, default="True")
parser.add_argument("--reference_model", type=str, default="True")
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--gradient_collection", type=str, default=False)
args = parser.parse_args()

if args.dataset_name == "all":
    dataset_names = ["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                     "pubmed_central", "wikipedia_(en)", "full_pile","WikiMIA64", "WikiMIA128","WikiMIA256",
                      "WikiMIAall"]
else:
    dataset_names = [args.dataset_name]
entropy_dict = {"12b":[], "6.9b":[],"1b":[], "410m":[]}
for dataset_name in dataset_names:
    member_entropy = np.load(f"entropy_online/{args.model_size}/{dataset_name}_member_entropy.npy")
    nonmember_entropy = np.load(f"entropy_online/{args.model_size}/{dataset_name}_nonmember_entropy.npy")

