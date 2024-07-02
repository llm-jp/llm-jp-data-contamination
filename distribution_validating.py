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
from scipy.stats import entropy, wasserstein_distance, ks_2samp
import argparse
def batched_data(dataset, batch_size):
    data_iter = iter(dataset)
    while True:
        batch = list(islice(data_iter, batch_size))
        if not batch:
            break
        yield batch

def figure_draw(data_dict, title, args):
    plt.figure(figsize=(10, 5))
    fig, axs = plt.subplots(len(data_dict), figsize=(10, 5 * len(data_dict)))
    axs = np.atleast_2d(axs)
    for ax, (dataset_name, dataset_loss) in zip(axs.flatten(), data_dict.items()):
        for phase_name, phase_loss in dataset_loss.items():
            weights = np.ones_like(phase_loss) / len(phase_loss)
            ax.hist(phase_loss, bins=50, label=phase_name, alpha=0.5, weights=weights)
        ax.set_title(f'{title} values histogram for {dataset_name} at {args.model_size} model')
        ax.set_xlabel(title)
        ax.set_ylabel('Percentage')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{title}_histograms_{args.model_size}.png")
    plt.show()



def loss_collection(model, dataset, args, batch_size=8):
    loss_collect = []
    prob_collect = []
    ppl_collect = []
    for batch in tqdm(batched_data(dataset, batch_size=batch_size)):
        tokenized_inputs = tokenizer([item for item in batch],
                                     return_tensors="pt",
                                     truncation=True,
                                     padding=True,
                                     max_length=2048)
        tokenized_inputs = {key: val.cuda(args.cuda) for key, val in tokenized_inputs.items()}
        target_labels = tokenized_inputs["input_ids"].clone()
        target_labels[tokenized_inputs["attention_mask"] == 0] = -100
        with torch.no_grad():
            outputs = model(**tokenized_inputs, labels=target_labels.cuda(args.cuda))
        loss, logits = outputs[:2]
        probabilities = torch.nn.functional.log_softmax(logits, dim=2)
        batch_size = tokenized_inputs["input_ids"].shape[0]
        seq_length = tokenized_inputs["input_ids"].shape[1]
        # 初始化
        all_prob = []
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


def js_divergence(dict, dataset_name):
    # Ensure p and q sum to 1
    js_matrix = np.zeros((3, 3))
    split_set = ["train", "valid", "test"]
    for idx1, set1 in enumerate(split_set):
        for idx2, set2 in enumerate(split_set):
            hist1, bin_edges = np.histogram(np.array(dict[dataset_name][set1]), bins=100, density=True)
            hist2, _ = np.histogram(np.array(dict[dataset_name][set2]), bins=100, density=True)
            eps = 1e-10
            hist1 += eps
            hist2 += eps
            # 确保向量总和为1（归一化），表示概率分布
            hist1 /= hist1.sum()
            hist2 /= hist2.sum()
            m = 0.5 * (hist1 + hist2)
            js_matrix[idx1][idx2] = 0.5 * entropy(hist1, m) + 0.5 * entropy(hist2, m)
    return js_matrix#close to zero means the two distributions are similar

def ks_hypothesis(dict, dataset_name):
    ks_matrix = np.zeros((3, 3))
    split_set = ["train", "valid", "test"]
    for idx1, set1 in enumerate(split_set):
        for idx2, set2 in enumerate(split_set):
            ks_stat, _ = ks_2samp(dict[dataset_name][set1], dict[dataset_name][set2])
            ks_matrix[idx1][idx2] = ks_stat
    return ks_matrix#close to zero means the two distributions are similar


#dataset_name = ["ArXiv", "DM Mathematics", "Enron Emails", "EuroParl", "FreeLaw", "Github", "Gutenberg (PG-19)",
#                "HackerNews", "NIH ExPorter", "PhilPapers", "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
#                "Ubuntu IRC", "USPTO Backgrounds", "Wikipedia (en)"]

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--model_size", type=str, default="70m")
parser.add_argument("--dataset_name", type=str, default="Pile-CC", choices=["ArXiv", "DM Mathematics", "Enron Emails", "EuroParl", "FreeLaw", "Github", "Gutenberg (PG-19)",
                "HackerNews", "NIH ExPorter", "PhilPapers", "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                "Ubuntu IRC", "USPTO Backgrounds", "Wikipedia (en)"])
parser.add_argument("--cuda", type=int, default=0, help="cuda device")
parser.add_argument("--skip_calculation", type=str, default="True")
args = parser.parse_args()

if args.skip_calculation == "True":
    skip_calculation = True
else:
    skip_calculation = False
if not skip_calculation:
    model = GPTNeoXForCausalLM.from_pretrained(
      f"EleutherAI/pythia-{args.model_size}-deduped",
      revision="step143000",
      cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
    ).half().eval()
    model = model.to_bettertransformer()
    model = model.cuda(args.cuda)
    tokenizer = AutoTokenizer.from_pretrained(
      f"EleutherAI/pythia-{args.model_size}-deduped",
      revision="step143000",
      cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
    )
    tokenizer.pad_token = tokenizer.eos_token
    loss_dict = {}
    prob_dict = {}
    ppl_dict = {}
    loss_dict[args.dataset_name] = {"train": [], "valid": [], "test": []}
    prob_dict[args.dataset_name] = {"train": [], "valid": [], "test": []}
    ppl_dict[args.dataset_name] = {"train": [], "valid": [], "test": []}
    for split in ["train", "valid", "test"]:
        if split in ["test", "valid"]:
            dataset = torch.load(f"by_dataset/{split}_{args.dataset_name}.pt")
            loss_list, prob_list, ppl_list = loss_collection(model, dataset, args, batch_size=args.batch_size)
            loss_dict[args.dataset_name][split].extend(loss_list)
            prob_dict[args.dataset_name][split].extend(prob_list)
            ppl_dict[args.dataset_name][split].extend(ppl_list)
        else:
            for i in range(1):
                dataset = torch.load(f"by_dataset/{split}_{args.dataset_name}_{i}.pt")
                loss_list, prob_list, ppl_list = loss_collection(model, dataset, args, batch_size=args.batch_size)
                loss_dict[args.dataset_name][split].extend(loss_list)
                prob_dict[args.dataset_name][split].extend(prob_list)
                ppl_dict[args.dataset_name][split].extend(ppl_list)
    pickle.dump(loss_dict, open(f"feature_result/{args.dataset_name}_{args.model_size}_loss_dict.pkl", "wb"))
    pickle.dump(prob_dict, open(f"feature_result/{args.dataset_name}_{args.model_size}_prob_dict.pkl", "wb"))
    pickle.dump(ppl_dict, open(f"feature_result/{args.dataset_name}_{args.model_size}_ppl_dict.pkl", "wb"))
loss_dict = pickle.load(open(f"feature_result/{args.dataset_name}_{args.model_size}_loss_dict.pkl", "rb"))
prob_dict = pickle.load(open(f"feature_result/{args.dataset_name}_{args.model_size}_prob_dict.pkl", "rb"))
ppl_dict = pickle.load(open(f"feature_result/{args.dataset_name}_{args.model_size}_ppl_dict.pkl", "rb"))
figure_draw(loss_dict, "Loss", args)
figure_draw(prob_dict, "Prob", args)
figure_draw(ppl_dict, "PPL", args)
for idx, dict in enumerate([loss_dict, prob_dict, ppl_dict]):
    if idx == 0:
        print("Loss Distribution Similarity Matrix")
    elif idx == 1:
        print("Prob Distribution Similarity Matrix")
    else:
        print("PPL Distribution Similarity Matrix")
    js_matrix = js_divergence(dict, args.dataset_name)
    print(js_matrix)
    ks_matrix = ks_hypothesis(dict, args.dataset_name)
    print(ks_matrix)


