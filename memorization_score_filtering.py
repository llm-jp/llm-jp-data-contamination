import pdb
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from utils import form_dataset, batched_data, clean_dataset
import argparse
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--model_size", type=str, default="410m")
parser.add_argument("--dataset_name", type=str, default="ArXiv", choices=["ArXiv", "DM Mathematics",
                 "FreeLaw", "Github",  "HackerNews", "NIH ExPorter",
                "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA", "all"])
parser.add_argument("--cuda", type=int, default=1, help="cuda device")
parser.add_argument("--skip_calculation", type=str, default="True")
parser.add_argument("--reference_model", type=str, default="True")
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--gradient_collection", type=str, default=False)
args = parser.parse_args()


if args.dataset_name == "all":
    dataset_names = ["ArXiv", "DM Mathematics",
                     "FreeLaw", "Github", "HackerNews", "NIH ExPorter",
                      "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                      "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA"]
    #dataset_names = ["Github", "HackerNews", "NIH ExPorter","Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
    #                "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA"]
else:
    dataset_names = [args.dataset_name]

model = GPTNeoXForCausalLM.from_pretrained(
    f"EleutherAI/pythia-{args.model_size}-deduped",
    revision="step143000",
    cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
    torch_dtype=torch.float16,
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
    dataset = form_dataset(dataset_name)
    device = f'cuda:{args.cuda}'
    mem_score = pandas.DataFrame(columns=["set_name", "batch_idx",  "mem_score"])
    for set_name in ["train", "test"]:
        cleaned_dataset = clean_dataset(dataset[set_name])
        for idx, batch in tqdm(enumerate(batched_data(cleaned_dataset, batch_size=args.batch_size))):
            if idx * args.batch_size > args.samples:
                break
            batched_text = [item for item in batch]
            tokenized_inputs = tokenizer(batched_text, return_tensors="pt", truncation=True, max_length=2048)
            tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
            #input_length = int(tokenized_inputs["input_ids"].shape[1] * ratio)
            #output_length = int(tokenized_inputs["input_ids"].shape[1] * (ratio + 0.1))
            input_length = 64
            output_length = 32
            generations = model.generate(tokenized_inputs["input_ids"][0][:input_length].unsqueeze(0),
                                         temperature=0.0, top_k=0, top_p=0, max_length=input_length+output_length,
                                         min_length=input_length+output_length)
            comparasion_result = generations["sequences"][0][input_length:] == tokenized_inputs["input_ids"][0][input_length:input_length+output_length]
            score = sum(comparasion_result) / (output_length - input_length)
            score = mem_score.cpu().numpy()
            mem_score = mem_score.append({"set_name": set_name, "batch_idx": idx, "mem_score": score}, ignore_index=True)
    mem_score.to_csv(f"{args.model_size}_{dataset_name}_mem_score.csv")
            #for idx, ratio in enumerate(np.linspace(0, 1, 11)[1:]):


