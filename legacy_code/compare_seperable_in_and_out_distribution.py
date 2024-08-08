import pickle
import argparse
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_length", type=int, default=2048)
parser.add_argument("--model_size", type=str, default="12b")
parser.add_argument("--dataset_name", type=str, default="Pile-CC", choices=["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                     "pubmed_central", "wikipedia_(en)", "full_pile", "all"])
parser.add_argument("--cuda", type=int, default=0, help="cuda device")
parser.add_argument("--refer_cuda", type=int, default=7, help="cuda device")
parser.add_argument("--skip_calculation", type=str, default="True")
parser.add_argument("--reference_model", type=str, default="True")
parser.add_argument("--samples", type=int, default=5000)
parser.add_argument("--gradient_collection", type=str, default=False)
parser.add_argument("--dir", type=str, default="feature_result_online")
args = parser.parse_args()

dataset_name = "wikipedia_(en)"
wiki_en_loss_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_loss_dict.pkl", "rb"))
wiki_en_prob_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_prob_dict.pkl", "rb"))
wiki_en_ppl_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_ppl_dict.pkl", "rb"))
wiki_en_mink_plus_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_mink_plus_dict.pkl", "rb"))
wiki_en_zlib_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_zlib_dict.pkl", "rb"))
wiki_en_refer_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_refer_dict.pkl", "rb"))
wiki_en_idx_list = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_idx_list.pkl", "rb"))
wiki_en_all_dict = [wiki_en_loss_dict, wiki_en_prob_dict, wiki_en_ppl_dict, wiki_en_mink_plus_dict, wiki_en_zlib_dict, wiki_en_refer_dict]
dataset_name = "WikiMIAall"
wiki_all_loss_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_loss_dict.pkl", "rb"))
wiki_all_prob_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_prob_dict.pkl", "rb"))
wiki_all_ppl_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_ppl_dict.pkl", "rb"))
wiki_all_mink_plus_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_mink_plus_dict.pkl", "rb"))
wiki_all_zlib_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_zlib_dict.pkl", "rb"))
wiki_all_refer_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_refer_dict.pkl", "rb"))
wiki_all_idx_list = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_idx_list.pkl", "rb"))
wiki_all_all_dict = [wiki_all_loss_dict, wiki_all_prob_dict, wiki_all_ppl_dict, wiki_all_mink_plus_dict, wiki_all_zlib_dict, wiki_all_refer_dict]

method_list = ["loss", "prob", "ppl", "mink_plus", "zlib", "refer"]
for idx, (method_name, wiki_en_dict, wiki_all_dict) in enumerate(zip(method_list,  wiki_en_all_dict, wiki_all_all_dict)):
    plt.figure(figsize=(12, 6))
    print(wiki_en_dict.keys())
    print(wiki_en_dict["wikipedia_(en)"].keys())
    for set_name in ["member", "nonmember"]:
#        weights = np.ones_like(wiki_en_dict["wikipedia_(en)"][set_name]) / len(wiki_en_dict["wikipedia_(en)"][set_name])
#        plt.hist(wiki_en_dict["wikipedia_(en)"][set_name], bins=50, label="wikien"+set_name, alpha=0.25, weights=weights)
#        weights = np.ones_like(wiki_all_dict["WikiMIAall"][set_name]) / len(wiki_all_dict["WikiMIAall"][set_name])
#        plt.hist(wiki_all_dict["WikiMIAall"][set_name], bins=50, label="wikiall"+set_name, alpha=0.25, weights=weights)
        sns.kdeplot(wiki_en_dict["wikipedia_(en)"][set_name], label="wikien"+set_name, alpha=0.75, bw_adjust=0.25, shade=True)
        sns.kdeplot(wiki_all_dict["WikiMIAall"][set_name], label="wikiall"+set_name, alpha=0.75, bw_adjust=0.25, shade=True)
    plt.title(f"{method_name} distribution comparison")
    plt.legend()
    plt.show()
