from datasets import load_dataset
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch.nn.functional as F
import argparse
import random
import seaborn as sns
from datasets import DatasetDict
import os
from torch.nn import CrossEntropyLoss
from utils import *



parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_length", type=int, default=2048)
parser.add_argument("--model_size", type=str, default="160m")
parser.add_argument("--dataset_name", type=str, default="Pile-CC", choices=["ArXiv", "DM Mathematics",
                 "FreeLaw", "Github",  "HackerNews", "NIH ExPorter",
                "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA", "all"])
parser.add_argument("--cuda", type=int, default=0, help="cuda device")
parser.add_argument("--refer_cuda", type=int, default=7, help="cuda device")
parser.add_argument("--skip_calculation", type=str, default="True")
parser.add_argument("--reference_model", type=str, default="True")
parser.add_argument("--samples", type=int, default=5000)
parser.add_argument("--dir", type=str, default="feature_result_online")
args = parser.parse_args()

if args.dataset_name == "all":
    # dataset_names = ["ArXiv", "DM Mathematics",
    #               "FreeLaw", "Github",  "HackerNews", "NIH ExPorter",
    #              "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
    #              "USPTO Backgrounds", "Wikipedia (en)","WikiMIA32","WikiMIA64", "WikiMIA128","WikiMIA256",
    #                  "WikiMIAall"]
    dataset_names = ["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                     "pubmed_central", "wikipedia_(en)", "full_pile"]
    # dataset_names = ["WikiMIA32","WikiMIA64", "WikiMIA128","WikiMIA256",
    #                   "WikiMIAall"]
else:
    dataset_names = [args.dataset_name]

for dataset_name in dataset_names:
    loss_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_loss_dict.pkl", "rb"))
    prob_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_prob_dict.pkl", "rb"))
    ppl_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_ppl_dict.pkl", "rb"))
    mink_plus_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_mink_plus_dict.pkl", "rb"))
    zlib_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_zlib_dict.pkl", "rb"))
    refer_dict = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_refer_dict.pkl", "rb"))
    idx_list = pickle.load(open(f"{args.dir}/{dataset_name}_{args.model_size}_idx_list.pkl", "rb"))
    all_dict = [loss_dict, prob_dict, ppl_dict, mink_plus_dict, zlib_dict, refer_dict]
    f = open(f"results/{dataset_name}_{args.model_size}_mixed_results.txt", "w")
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for idx, dict in enumerate(all_dict):
            if idx == 0:
                train_data, test_data, combine_data = mix_distribution(loss_dict, dataset_name, "Loss", args, ratio=ratio)
                print("Loss Distribution Similarity Matrix")
                f.write("Loss Distribution Similarity Matrix\n")
            elif idx == 1:
                train_data, test_data, combine_data =mix_distribution(prob_dict, dataset_name, "Prob", args, ratio=ratio)
                print("Prob Distribution Similarity Matrix")
                f.write("Prob Distribution Similarity Matrix\n")
            elif idx == 2:
                train_data, test_data, combine_data =mix_distribution(ppl_dict, dataset_name, "PPL", args, ratio=ratio)
                print("PPL Distribution Similarity Matrix")
                f.write("PPL Distribution Similarity Matrix\n")
            elif idx == 3:
                train_data, test_data, combine_data =mix_distribution(mink_plus_dict, dataset_name, "Mink_plus", args, ratio=ratio)
                f.write("Mink_plus Distribution Similarity Matrix\n")
                print("Mink_plus Distribution Similarity Matrix")
            elif idx == 4:
                train_data, test_data, combine_data =mix_distribution(zlib_dict, dataset_name, "Zlib", args, ratio=ratio)
                print("Zlib Distribution Similarity Matrix")
                f.write("Zlib Distribution Similarity Matrix\n")
            elif idx == 5:
                residual_dict = {}
                residual_dict[dataset_name] = {"train": [], "valid": [], "test": []}
                for split in ["train", "valid", "test"]:
                    residual_dict[dataset_name][split] = [
                        loss_dict[dataset_name][split][i] - refer_dict[dataset_name][split][i]
                        for i in range(len(loss_dict[dataset_name][split]))]
                train_data, test_data, combine_data = mix_distribution(residual_dict, dataset_name, "Refer", args, ratio=ratio)
                print("Refer Distribution Similarity Matrix")
                f.write("Refer Distribution Similarity Matrix\n")
            print(idx)
            new_dict = {}
            new_dict[dataset_name] = {"train": train_data, "merged": combine_data, "test": test_data}
            calculate_mean_var(new_dict, dataset_name, split_set=["train", "merged", "test"])
            js_matrix = js_divergence(dict, dataset_name)
            print(js_matrix)
            f.write("JS Divergence Matrix\n")
            f.write(str(js_matrix) + '\n')
            ks_matrix, ks_p_value_matrix = ks_hypothesis(new_dict, dataset_name, split_set=["train", "merged", "test"])
            print(ks_matrix)
            f.write("KS Hypothesis Matrix\n")
            f.write(str(ks_matrix) + '\n')
            print(ks_p_value_matrix)
            f.write("KS P Value Matrix\n")
            f.write(str(ks_p_value_matrix) + '\n')
            ws_matrix = wasserstein_distance_caculate(new_dict, dataset_name, split_set=["train", "merged", "test"])
            print(ws_matrix)
            f.write("Wasserstein Distance Matrix\n")
            f.write(str(ws_matrix) + '\n')
    f.close()