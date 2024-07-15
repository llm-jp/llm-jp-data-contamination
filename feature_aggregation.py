from datasets import load_dataset
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
import pickle
from itertools import islice
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch.nn.functional as F
from scipy.stats import entropy, ks_2samp, kurtosis, wasserstein_distance
import argparse
import random
import seaborn as sns
import zlib
from datasets import DatasetDict
import os

dataset_name = "Pile-CC"
model_size = "160m"
loss_dict = pickle.load(open(f"feature_result/{dataset_name}_{model_size}_loss_dict.pkl", "rb"))
prob_dict = pickle.load(open(f"feature_result/{dataset_name}_{model_size}_prob_dict.pkl", "rb"))
ppl_dict = pickle.load(open(f"feature_result/{dataset_name}_{model_size}_ppl_dict.pkl", "rb"))
mink_plus_dict = pickle.load(open(f"feature_result/{dataset_name}_{model_size}_mink_plus_dict.pkl", "rb"))
zlib_dict = pickle.load(open(f"feature_result/{dataset_name}_{model_size}_zlib_dict.pkl", "rb"))

