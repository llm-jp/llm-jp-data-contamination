import pdb
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from utils import form_dataset, batched_data_with_indices, clean_dataset
import argparse
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--model_size", type=str, default="410m")
parser.add_argument("--max_length", type=int, default=96)
parser.add_argument("--dataset_name", type=str, default="all", choices=["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
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
    # dataset_names = ["WikiMIA64", "WikiMIA128","WikiMIA256",
    #                  "WikiMIAall"]
else:
    dataset_names = [args.dataset_name]

model = GPTNeoXForCausalLM.from_pretrained(
    f"EleutherAI/pythia-{args.model_size}-deduped",
    revision="step143000",
    cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
    torch_dtype=torch.bfloat16,
    # attn_implementation="sdpa"
).cuda(args.cuda).eval()
tokenizer = AutoTokenizer.from_pretrained(
  f"EleutherAI/pythia-{args.model_size}-deduped",
  revision="step143000",
  cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
)
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.generation_config.output_hidden_states = True
#model.generation_config.output_attentions = True
model.generation_config.output_scores = True
model.generation_config.return_dict_in_generate = True


for dataset_name in dataset_names:
    dataset = load_dataset("iamgroot42/mimir", dataset_name,
                           split="ngram_13_0.2") if dataset_name != "full_pile" else load_dataset("iamgroot42/mimir",
                                                                                                  "full_pile",
                                                                                                  split="none")
    device = f'cuda:{args.cuda}'
    mem_score = pandas.DataFrame(columns=["set_name", "original_idx",  "mem_score"])
    for set_name in ["member", "nonmember"]:
        cleaned_data, orig_indices = clean_dataset(dataset[set_name], dataset_name, online=True)
        for idx, (data_batch, orig_indices_batch) in tqdm(enumerate(batched_data_with_indices(cleaned_data, orig_indices, batch_size=args.batch_size))):
            orig_idx = [item for item in orig_indices_batch]
            if idx * args.batch_size > args.samples:
                break
            batched_text = [item for item in data_batch]
            tokenized_inputs = tokenizer(batched_text, return_tensors="pt", truncation=True, max_length=args.max_length)
            tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
            #input_length = int(tokenized_inputs["input_ids"].shape[1] * ratio)
            #output_length = int(tokenized_inputs["input_ids"].shape[1] * (ratio + 0.1))
            input_length = 32
            output_length = 32
            if tokenized_inputs["input_ids"][0].shape[0] < input_length + output_length:
                input_length = tokenized_inputs["input_ids"][0].shape[0] - output_length
                if input_length < 0:
                    continue
            generations = model.generate(tokenized_inputs["input_ids"][0][:input_length].unsqueeze(0),
                                         temperature=0.0, top_k=0, top_p=0, max_length=input_length+output_length,
                                         min_length=input_length+output_length)
            comparasion_result = generations["sequences"][0][input_length:] == tokenized_inputs["input_ids"][0][input_length:input_length+output_length]
            score = sum(comparasion_result) / output_length
            score = score.cpu().numpy()
            mem_score = mem_score._append({"set_name": set_name, "original_idx": orig_idx[0], "mem_score": score}, ignore_index=True)
    os.makedirs(f"mem_score_online/{args.model_size}", exist_ok=True)
    mem_score.to_csv(f"mem_score_online/{args.model_size}/{dataset_name}_mem_score.csv")
            #for idx, ratio in enumerate(np.linspace(0, 1, 11)[1:]):

