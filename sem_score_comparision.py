import pdb
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from utils import form_dataset, batched_data_with_indices, clean_dataset
import argparse
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pandas
import os
from datasets import load_dataset
import copy
import evaluate
import pdb
import pickle

def bleurt_score(predictions, references):
    """Compute the average BLEURT score over the gpt responses

    Args:
        predictions (list): List of gpt responses generated by GPT-3.5 under a certain instruction
        references (list): List of sentence 2 collected from raw dataset (e.g. wnli)

    Returns:
        bluert_scores: List of bluert scores
    """
    bluert_scores = bleurt.compute(predictions=predictions,
                                   references=references)['scores']
    return bluert_scores


def rougeL_score(predictions, references):
    """Compute the rougel score over the gpt responses

    Args:
        predictions (list): List of gpt responses generated by GPT-3.5 under a certain instruction
        references (list): List of sentence 2 collected from raw dataset (e.g. wnli)

    Returns:
        rougeL_scores: List of rougeL scores
    """
    rougeL_score = rouge.compute(predictions=predictions,
                                 references=references, use_aggregator=False)['rougeL']
    return rougeL_score

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--model_size", type=str, default="410m")
parser.add_argument("--dataset_name", type=str, default="all", choices=["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                     "pubmed_central", "wikipedia_(en)", "full_pile","WikiMIA64", "WikiMIA128","WikiMIA256",
                      "WikiMIAall"])
parser.add_argument("--cuda", type=int, default=1, help="cuda device")
parser.add_argument("--skip_calculation", type=str, default="True")
parser.add_argument("--reference_model", type=str, default="True")
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--generation_samples", type=int, default=15)
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

bleurt =  evaluate.load('bleurt', 'bleurt-20', model_type="metric")
rouge = evaluate.load('rouge')

for input_length in [48]:
    temp_input_length = copy.deepcopy(input_length)
    for dataset_name in dataset_names:
        if "WikiMIA" in dataset_name:
            dataset = form_dataset(dataset_name)
            dataset["member"] = dataset["train"]
            dataset["nonmember"] = dataset["test"]
        else:
            dataset = load_dataset("iamgroot42/mimir", dataset_name,
                               split="ngram_13_0.2") if dataset_name != "full_pile" else load_dataset("iamgroot42/mimir",
                                                                                                      "full_pile",
                                                                                                      split="none")
        device = f'cuda:{args.cuda}'
        mem_score = pandas.DataFrame(columns=["set_name", "original_idx",  "mem_score"])
        generation_samples_list = []
        for set_name in ["member", "nonmember"]:
            cleaned_data, orig_indices = clean_dataset(dataset[set_name], dataset_name, online=True)
            for idx, (data_batch, orig_indices_batch) in tqdm(enumerate(batched_data_with_indices(cleaned_data, orig_indices, batch_size=args.batch_size))):
                orig_idx = [item for item in orig_indices_batch]
                if idx * args.batch_size > args.samples:
                    break
                batched_text = [item for item in data_batch]
                tokenized_inputs = tokenizer(batched_text, return_tensors="pt", truncation=True, padding=True)
                tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
                #input_length = int(tokenized_inputs["input_ids"].shape[1] * ratio)
                #output_length = int(tokenized_inputs["input_ids"].shape[1] * (ratio + 0.1))
                temp_results = []
                full_decoded = []
                for _ in tqdm(range(args.generation_samples)):
                    generations = model.generate(tokenized_inputs["input_ids"][0][:input_length].unsqueeze(0),
                                                 do_sample=True,
                                                 temperature=1,
                                                 max_length=1024,  # input+output
                                                 top_k=50,
                                                 top_p=1,
                                                )
                    temp_results.append(tokenizer.decode(generations["sequences"][0][input_length:]))
                    full_decoded.append(tokenizer.decode(generations["sequences"][0]))
                text_to_compare = tokenizer.decode(tokenized_inputs["input_ids"][0][input_length:])
                #pdb.set_trace()
                bleurt_scores = bleurt_score(temp_results, [text_to_compare for _ in range(args.generation_samples)])
                rougeL_scores = rougeL_score(temp_results, [text_to_compare for _ in range(args.generation_samples)])
                mem_score = mem_score._append({"set_name": set_name, "original_idx": orig_idx[0],
                                               "bleurt_score": bleurt_scores, "rougle_scores":rougeL_scores
                                               }, ignore_index=True)
                input_length = temp_input_length
                generation_samples_list.append([batched_text[0], full_decoded, temp_results, text_to_compare])
        os.makedirs(f"sem_mem_score_online/{args.model_size}", exist_ok=True)
        mem_score.to_csv(f"sem_mem_score_online/{args.model_size}/{dataset_name}_{temp_input_length}_sem_mem_score.csv")
        pickle.dump(generation_samples_list, open(f"sem_mem_score_online/{args.model_size}/{dataset_name}_{temp_input_length}_generation_samples.pkl", "wb"))

