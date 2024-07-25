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
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--model_size", type=str, default="160m")
parser.add_argument("--dataset_name", type=str, default="Pile-CC", choices=["ArXiv", "DM Mathematics",
                 "FreeLaw", "Github",  "HackerNews", "NIH ExPorter",
                "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA", "all"])
parser.add_argument("--cuda", type=int, default=0, help="cuda device")
parser.add_argument("--refer_cuda", type=int, default=7, help="cuda device")
parser.add_argument("--reference_model", type=str, default=False)
parser.add_argument("--samples", type=int, default=5000)
parser.add_argument("--dir", type=str, default="mem_score_result")
args = parser.parse_args()

if args.dataset_name == "all":
    dataset_names = ["ArXiv", "DM Mathematics",
                  "FreeLaw", "Github",  "HackerNews", "NIH ExPorter",
                 "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                 "USPTO Backgrounds", "Wikipedia (en)","WikiMIA32","WikiMIA64", "WikiMIA128","WikiMIA256",
                     "WikiMIAall"]
else:
    dataset_names = [args.dataset_name]
for dataset_name in dataset_names:
    mem_score_data = pd.read_csv(f"mem_score/{args.model_size}/{dataset_name}_mem_score.csv", index_col=0)
    mem_score_data["original_idx"] = mem_score_data["original_idx"].astype(int)
    train_dict = mem_score_data[mem_score_data['set_name'] == 'train'].set_index('original_idx')['mem_score'].to_dict()
    test_dict = mem_score_data[mem_score_data['set_name'] == 'test'].set_index('original_idx')['mem_score'].to_dict()
    valid_dict = mem_score_data[mem_score_data['set_name'] == 'test'].set_index('original_idx')['mem_score'].to_dict()
    print(sum(mem_score_data[mem_score_data["set_name"] == "test"]["mem_score"] == -1))
    print(sum(mem_score_data[mem_score_data["set_name"] == "train"]["mem_score"] == -1))
    print(sum(mem_score_data[mem_score_data["set_name"] == "valid"]["mem_score"] == -1))
