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

def batched_data(dataset, batch_size):
    data_iter = iter(dataset)
    while True:
        batch = list(islice(data_iter, batch_size))
        if not batch:
            break
        yield batch

def figure_draw(data_dict, title):
    plt.figure(figsize=(10, 5))
    fig, axs = plt.subplots(len(data_dict), figsize=(10, 5 * len(data_dict)))
    axs = np.atleast_2d(axs)
    for ax, (dataset_name, dataset_loss) in zip(axs.flatten(), data_dict.items()):
        for phase_name, phase_loss in dataset_loss.items():
            weights = np.ones_like(phase_loss) / len(phase_loss)
            ax.hist(phase_loss, bins=50, label=phase_name, alpha=0.5, weights=weights)
        ax.set_title(f'{title} values histogram for {dataset_name}')
        ax.set_xlabel(title)
        ax.set_ylabel('Percentage')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{title}_histograms.png")
    plt.show()



def loss_collection(model, dataset, batch_size=8):
    loss_collect = []
    prob_collect = []
    ppl_collect = []
    for batch in tqdm(batched_data(dataset, batch_size=batch_size)):
        tokenized_inputs = tokenizer([item for item in batch],
                                     return_tensors="pt",
                                     truncation=True,
                                     padding=True,
                                     max_length=2048)
        tokenized_inputs = {key: val.to("cuda") for key, val in tokenized_inputs.items()}
        target_labels = tokenized_inputs["input_ids"].clone()
        target_labels[tokenized_inputs["attention_mask"] == 0] = -100
        with torch.no_grad():
            outputs = model(**tokenized_inputs, labels=target_labels.cuda())
        loss, logits = outputs[:2]
        probabilities = torch.nn.functional.log_softmax(logits, dim=2)
        batch_size = tokenized_inputs["input_ids"].shape[0]
        seq_length = tokenized_inputs["input_ids"].shape[1]
        # 初始化
        all_prob = []
        prob_collect = []
        ppl_collect = []

        # 获取每个样本的概率
        for idx in range(batch_size):
            logits_i = logits[i].unsqueeze(0)  # Shape (1, seq_length, vocab_size)
            target_i = target_labels[i].unsqueeze(0)  # Shape (1, seq_length)
            shift_logits = logits_i[:, :-1, :].contiguous()
            shift_labels = target_i[:, 1:].contiguous()
            # 计算交叉熵损失并移除填充 token 贡献
            loss_i = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                                     reduction='none')
            # Create a mask to ignore the loss from padding tokens
            valid_mask = shift_labels != -100
            # 只有有效的 token 计算损失
            loss_i = loss_i * valid_mask.view(-1)
            # 计算每个样本的平均损失
            loss_i = loss_i.sum() / valid_mask.sum()
            loss_collect.append(loss_i.item())
            input_ids_processed = tokenized_inputs["input_ids"][idx]
            attention_mask_processed = tokenized_inputs["attention_mask"][idx]
            probs = probabilities[idx]  # 形状为 (seq_length, vocab_size)
            # 使用 attention_mask 筛选有效的 token
            valid_probs = probs[attention_mask_processed == 1]
            valid_token_ids = input_ids_processed[attention_mask_processed == 1]
            # 获取这些有效 token 的概率
            selected_probs = valid_probs[np.arange(valid_token_ids.shape[0]), valid_token_ids]
            # 计算 topk 概率
            k_length = int(len(selected_probs) * 0.2)
            topk_prob = np.sort(selected_probs.cpu().numpy())[:k_length]
            pred = -np.mean(topk_prob).item()
            # perplexity's value
            ppl = torch.exp(loss).item()
            # 收集结果
            all_prob.append(selected_probs.cpu().numpy())
            prob_collect.append(pred)
            ppl_collect.append(ppl)
    return loss_collect, prob_collect, ppl_collect

#dataset_name = ["ArXiv", "DM Mathematics", "Enron Emails", "EuroParl", "FreeLaw", "Github", "Gutenberg (PG-19)",
#                "HackerNews", "NIH ExPorter", "PhilPapers", "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
#                "Ubuntu IRC", "USPTO Backgrounds", "Wikipedia (en)"]
dataset_name = ["Pile-CC"]
split_name = ["train", "valid", "test"]

model_size = "70m"
model = GPTNeoXForCausalLM.from_pretrained(
  f"EleutherAI/pythia-{model_size}-deduped",
  revision="step143000",
  cache_dir=f"./pythia-{model_size}-deduped/step143000",
).half().eval()
model = model.to_bettertransformer()
model = model.cuda()
tokenizer = AutoTokenizer.from_pretrained(
  f"EleutherAI/pythia-{model_size}-deduped",
  revision="step143000",
  cache_dir=f"./pythia-{model_size}-deduped/step143000",
)
tokenizer.pad_token = tokenizer.eos_token
loss_dict = {}
prob_dict = {}
ppl_dict = {}
for name in dataset_name:
    loss_dict[name] = {"train": [], "valid": [], "test": []}
    prob_dict[name] = {"train": [], "valid": [], "test": []}
    ppl_dict[name] = {"train": [], "valid": [], "test": []}
    for split in split_name:
        if split in ["test", "valid"]:
            dataset = torch.load(f"by_dataset/{split}_{name}.pt")
            loss_list, prob_list, ppl_list = loss_collection(model, dataset)
            loss_dict[name][split].extend(loss_list)
            prob_dict[name][split].extend(prob_list)
            ppl_dict[name][split].extend(ppl_list)
        else:
            for i in range(1):
                dataset = torch.load(f"by_dataset/{split}_{name}_{i}.pt")
                loss_list, prob_list, ppl_list = loss_collection(model, dataset)
                loss_dict[name][split].extend(loss_list)
                prob_dict[name][split].extend(prob_list)
                ppl_dict[name][split].extend(ppl_list)
pickle.dump(loss_dict, open("loss_dict.pkl", "wb"))
pickle.dump(prob_dict, open("prob_dict.pkl", "wb"))
pickle.dump(ppl_dict, open("ppl_dict.pkl", "wb"))
figure_draw(loss_dict, "Loss")
figure_draw(prob_dict, "Prob")
figure_draw(ppl_dict, "PPL")


