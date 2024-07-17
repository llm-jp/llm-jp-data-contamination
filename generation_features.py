import pdb
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from utils import form_dataset, batched_data, clean_dataset
import argparse
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--model_size", type=str, default="160m")
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

skip_calculation = False
model = GPTNeoXForCausalLM.from_pretrained(
  f"EleutherAI/pythia-{args.model_size}-deduped",
  revision="step143000",
  cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
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
model.generation_config.output_attentions = True
model.generation_config.output_scores = True
model.generation_config.return_dict_in_generate = True


f = open(f"{args.model_size}_embedding_result.txt", "w")
for dataset_name in dataset_names:
    dataset = form_dataset(dataset_name)
    device = f'cuda:{args.cuda}'
    member_embed_list = {}
    non_member_embed_list = {}
    for set_name in ["train", "test"]:
        cleaned_dataset = clean_dataset(dataset)
        member_entropy = []
        non_member_entropy = []
        for idx, batch in tqdm(enumerate(batched_data(dataset[set_name], batch_size=args.batch_size))):
            if idx * args.batch_size > args.samples:
                break
            batched_text = [item for item in batch]
            tokenized_inputs = tokenizer(batched_text, return_tensors="pt", truncation=True, max_length=2048)
            tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
            local_entropy = []
            for idx, ratio in enumerate(np.linspace(0, 1, 21)[1:]):
                input_length = int(tokenized_inputs["input_ids"].shape[1]*ratio)
                output_length = int(tokenized_inputs["input_ids"].shape[1]*(ratio+0.05))
                generations = model.generate(tokenized_inputs["input_ids"][0][:input_length].unsqueeze(0),temperature=0.0,top_k=0, top_p=0, max_length=output_length,min_length=output_length)
                logits = generations["scores"]
                #pdb.set_trace()
                probability_scores = torch.nn.functional.softmax(logits[0].float(), dim=1)
                entropy_scores = torch.distributions.Categorical(probs=probability_scores).entropy()
                local_entropy.append(entropy_scores.cpu().item())
            if set_name == "train":
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

# x轴的值
x = np.linspace(0, 1, 21)[1:]

# 创建图
plt.figure(figsize=(10, 6))

# 绘制member_entropy的均值和方差
plt.plot(x, mean_member, label='Member Entropy', color='blue')
plt.fill_between(x, mean_member - std_member, mean_member + std_member, color='blue', alpha=0.2)

# 绘制non_member_entropy的均值和方差
plt.plot(x, mean_non_member, label='Non-Member Entropy', color='red')
plt.fill_between(x, mean_non_member - std_non_member, mean_non_member + std_non_member, color='red', alpha=0.2)

# 添加图例和标签
plt.xlabel('Value')
plt.ylabel('Entropy')
plt.title('Mean and Variance of Entropy')
plt.legend()
plt.grid(True)
plt.show()



