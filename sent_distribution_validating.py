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
parser.add_argument("--dataset_name", type=str, default="Pile-CC", choices=["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                     "pubmed_central", "wikipedia_(en)", "full_pile", "all", "ArXiv"])
parser.add_argument("--cuda", type=int, default=0, help="cuda device")
parser.add_argument("--refer_cuda", type=int, default=7, help="cuda device")
parser.add_argument("--min_len", type=int, default=50)
parser.add_argument("--local_data", type=str, default="True")
parser.add_argument("--samples", type=int, default=5000)
parser.add_argument("--gradient_collection", type=str, default=False)
parser.add_argument("--dir", type=str, default="filtered_result")
args = parser.parse_args()

dataset_names = get_dataset_list(args.dataset_name)

model = GPTNeoXForCausalLM.from_pretrained(
  f"EleutherAI/pythia-{args.model_size}-deduped",
  revision="step143000",
  cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
  torch_dtype=torch.bfloat16,
).cuda(args.cuda).eval()
#model = model.to_bettertransformer()
tokenizer = AutoTokenizer.from_pretrained(
  f"EleutherAI/pythia-{args.model_size}-deduped",
  revision="step143000",
  cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
)

refer_model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-base-alpha-3b-v2",
                                                   trust_remote_code=True,
                                                   torch_dtype=torch.bfloat16).cuda(args.refer_cuda).eval()
refer_tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-3b-v2")
tokenizer.pad_token = tokenizer.eos_token
refer_tokenizer.pad_token = refer_tokenizer.eos_token
df = pd.DataFrame()
for dataset_name in dataset_names:
    dataset = obtain_dataset(dataset_name, min_len=args.min_len, local_data=True if args.local_data == "True" else False)
    loss_dict = {}
    prob_dict = {}
    ppl_dict = {}
    mink_plus_dict = {}
    zlib_dict = {}
    refer_dict = {}
    grad_dict = {}
    loss_dict[dataset_name] = {"member": [], "nonmember": []}
    prob_dict[dataset_name] = {"member": [], "nonmember": []}
    ppl_dict[dataset_name] = {"member": [], "nonmember": []}
    mink_plus_dict[dataset_name] = {"member": [], "nonmember": []}
    zlib_dict[dataset_name] = {"member": [], "nonmember": []}
    refer_dict[dataset_name] = {"member": [], "nonmember": []}
    grad_dict[dataset_name] = {"member": [], "nonmember": []}
    for split in ["member", "nonmember"]:
        loss_list, prob_list, ppl_list, mink_plus_list, zlib_list, refer_list, idx_list = feature_collection(model, tokenizer, dataset[split], args,
                                                                                                             dataset_name,
                                                                                       min_len = args.min_len,
                                                                                       upper_limit=args.samples,
                                                                                       refer_model=refer_model,
                                                                                       refer_tokenizer=refer_tokenizer,
                                                                                       )
        loss_dict[dataset_name][split].extend(loss_list)
        prob_dict[dataset_name][split].extend(prob_list)
        ppl_dict[dataset_name][split].extend(ppl_list)
        mink_plus_dict[dataset_name][split].extend(mink_plus_list)
        zlib_dict[dataset_name][split].extend(zlib_list)
        refer_dict[dataset_name][split].extend(refer_list)
        grad_dict[dataset_name][split].extend(idx_list)
    os.makedirs(args.dir, exist_ok=True)
    pickle.dump(loss_dict, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_loss_dict.pkl", "wb"))
    pickle.dump(prob_dict, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_prob_dict.pkl", "wb"))
    pickle.dump(ppl_dict, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_ppl_dict.pkl", "wb"))
    pickle.dump(mink_plus_dict, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_mink_plus_dict.pkl", "wb"))
    pickle.dump(zlib_dict, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_zlib_dict.pkl", "wb"))
    pickle.dump(refer_dict, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_refer_dict.pkl", "wb"))
    pickle.dump(grad_dict, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_grad_dict.pkl", "wb"))
    pickle.dump(idx_list, open(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}_idx_list.pkl", "wb"))
    df = results_caculate_and_draw(dataset_name, args, df, split_set=["member", "nonmember"])
df.to_csv(f"{args.dir}/{dataset_name}/{args.min_len}_{args.model_size}.csv")



