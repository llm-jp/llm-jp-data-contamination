import pdb
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation import LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList, MaxLengthCriteria
from utils import form_dataset, batched_data_with_indices, clean_dataset
import argparse
from tqdm import tqdm
import torch
import torch.quantization
import matplotlib.pyplot as plt
import pandas
import os
from datasets import load_dataset
import copy
import pickle
from utils import obtain_dataset, get_dataset_list

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
parser.add_argument("--generation_samples", type=int, default=10)
args = parser.parse_args()



dataset_names = get_dataset_list("WikiMIA")
bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # 开启8位量化
        bnb_8bit_use_double_quant=True,  # 使用双重量化技术
        bnb_8bit_compute_dtype=torch.float16  # 计算过程中使用float16
    )
model = GPTNeoXForCausalLM.from_pretrained(
    f"EleutherAI/pythia-{args.model_size}-deduped",
    revision="step143000",
    cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
).cuda(args.cuda).eval()
tokenizer = AutoTokenizer.from_pretrained(
  f"EleutherAI/pythia-{args.model_size}-deduped",
  revision="step143000",
  cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
)
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = model.generation_config.eos_token_id

model.generation_config.return_dict_in_generate = True

for input_length in [48]:
    temp_input_length = copy.deepcopy(input_length)
    for dataset_name in dataset_names:
        dataset = obtain_dataset(dataset_name)
        device = f'cuda:{args.cuda}'
        generation_samples_list = []
        for set_name in ["member", "nonmember"]:
            cleaned_data, orig_indices = clean_dataset(dataset[set_name])
            for idx, (data_batch, orig_indices_batch) in tqdm(enumerate(batched_data_with_indices(cleaned_data, orig_indices, batch_size=args.batch_size))):
                orig_idx = [item for item in orig_indices_batch]
                if idx * args.batch_size > args.samples:
                    break
                batched_text = [item for item in data_batch]
                tokenized_inputs = tokenizer(batched_text, return_tensors="pt", truncation=True, padding=True,
                                             max_length=1024)
                tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
                temp_results = []
                full_decoded = []
                for _ in tqdm(range(args.generation_samples)):
                    if tokenized_inputs["input_ids"].shape[1] <= input_length:
                        generations = model.generate(tokenized_inputs["input_ids"][0][:int(tokenized_inputs["input_ids"].shape[1]/2)].unsqueeze(0),
                                                     do_sample=True,
                                                     temperature=1,
                                                     max_length=len(tokenized_inputs["input_ids"][0]),  # input+output
                                                     top_k=50,
                                                     top_p=1,
                                                     )
                    else:
                        generations = model.generate(tokenized_inputs["input_ids"][0][:input_length].unsqueeze(0),
                                                     do_sample=True,
                                                     temperature=1,
                                                     max_length=len(tokenized_inputs["input_ids"][0]),  # input+output
                                                     top_k=50,
                                                     top_p=1,
                                                    )
                    temp_results.append(tokenizer.decode(generations["sequences"][0][input_length:]))
                    full_decoded.append(tokenizer.decode(generations["sequences"][0]))
                text_to_compare = tokenizer.decode(tokenized_inputs["input_ids"][0][input_length:])
                input_length = temp_input_length
                generation_samples_list.append([batched_text[0], full_decoded, temp_results, text_to_compare])
        os.makedirs(f"sem_mem_score_online/{args.model_size}", exist_ok=True)
        pickle.dump(generation_samples_list, open(f"sem_mem_score_online/{args.model_size}/{dataset_name}_{temp_input_length}_generation_samples.pkl", "wb"))


