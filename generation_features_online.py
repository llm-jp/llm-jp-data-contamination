import pdb
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from utils import form_dataset, clean_dataset, batched_data_with_indices
import argparse
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import copy
import os

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
    # dataset_names = ["WikiMIA64", "WikiMIA128","WikiMIA256",
    #                   "WikiMIAall"]
else:
    dataset_names = [args.dataset_name]

skip_calculation = False
model = GPTNeoXForCausalLM.from_pretrained(
  f"EleutherAI/pythia-{args.model_size}-deduped",
  revision="step143000",
  cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
   torch_dtype=torch.bfloat16,
    #attn_implementation="sdpa"
).cuda(args.cuda).eval()
#model = model.to_bettertransformer()

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
    if "WikiMIA" in dataset_name:
        dataset = form_dataset(dataset_name)
        dataset["member"] = dataset["train"]
        dataset["nonmember"] = dataset["test"]
    elif "temporal_arxiv" in dataset_name:
        dataset = load_dataset("iamgroot42/mimir", dataset_name,
                               split="2023_06")
    else:
        dataset = load_dataset("iamgroot42/mimir", dataset_name,
                               split="ngram_13_0.2") if dataset_name != "full_pile" else load_dataset(
            "iamgroot42/mimir",
            "full_pile",
            split="none")
    device = f'cuda:{args.cuda}'
    member_entropy = []
    non_member_entropy = []
    for set_name in ["member", "nonmember"]:
        cleaned_data, orig_indices = clean_dataset(dataset[set_name], dataset_name, online=True)
        for idx, (data_batch, orig_indices_batch) in tqdm(enumerate(batched_data_with_indices(cleaned_data, orig_indices, batch_size=args.batch_size))):
            if idx * args.batch_size > args.samples:
                break
            orig_idx = [item for item in orig_indices_batch]
            batched_text = [item for item in data_batch]
            tokenized_inputs = tokenizer(batched_text, return_tensors="pt", truncation=True,
                                         )
            tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
            if tokenized_inputs["input_ids"][0].shape[0] < 100:
                    continue
            local_entropy = []
            for input_length in list(range(1,65)):
                generations = model.generate(tokenized_inputs["input_ids"][0][:input_length].unsqueeze(0),temperature=0.0,top_k=0, top_p=0, max_length=input_length+1,min_length=input_length+1)
                logits = torch.stack(generations["scores"]).squeeze()
                #pdb.set_trace()
                probability_scores = torch.nn.functional.softmax(logits.float(), dim=0)
                entropy_scores = torch.distributions.Categorical(probs=probability_scores).entropy().mean()
                local_entropy.append(entropy_scores.cpu().item())
            if set_name == "member":
                member_entropy.append(local_entropy)
            else:
                non_member_entropy.append(local_entropy)
    member_entropy = np.array(member_entropy)
    non_member_entropy = np.array(non_member_entropy)
    # 计算均值和方差
    mean_member = np.mean(member_entropy, axis=0)
    std_member = np.std(member_entropy, axis=0)

    mean_non_member = np.mean(non_member_entropy, axis=0)
    std_non_member = np.std(non_member_entropy, axis=0)
    os.makedirs(f"entropy_online/{args.model_size}", exist_ok=True)
    # x轴的值
    x = list(range(1,65))

    # 创建图
    plt.figure(figsize=(10, 6))

    # 绘制member_entropy的均值和方差
    plt.plot(x, mean_member, label='Member Entropy', color='blue')
    plt.fill_between(x, mean_member - std_member, mean_member + std_member, color='blue', alpha=0.2)

    # 绘制non_member_entropy的均值和方差
    plt.plot(x, mean_non_member, label='Non-Member Entropy', color='red')
    plt.fill_between(x, mean_non_member - std_non_member, mean_non_member + std_non_member, color='red', alpha=0.2)
    np.save(f"entropy_online/{args.model_size}/{dataset_name}_member_entropy.npy", member_entropy)
    np.save(f"entropy_online/{args.model_size}/{dataset_name}_non_member_entropy.npy", non_member_entropy)

    # 添加图例和标签
    plt.xlabel('Value')
    plt.ylabel('Entropy')
    plt.title('Mean and Variance of Entropy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'entropy_{dataset_name}_{args.model_size}.png')
    plt.show()



