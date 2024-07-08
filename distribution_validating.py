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
from scipy.stats import entropy, wasserstein_distance, ks_2samp, kurtosis
import argparse
import random
import seaborn as sns
import zlib
from datasets import DatasetDict

def batched_data(dataset, batch_size):
    data_iter = iter(dataset)
    while True:
        batch = list(islice(data_iter, batch_size))
        if not batch:
            break
        yield batch

def mix_distribution(dict, dataset_name, title, args, ratio=0.8, total_num=10000):
    train_data = dict[dataset_name]["train"]
    test_data = dict[dataset_name]["test"]
    train_data_num = total_num*ratio
    test_data_num = total_num*(1-ratio)
    train_data = random.sample(train_data, min(int(train_data_num), len(train_data)))
    test_data = random.sample(test_data, min(int(test_data_num), len(test_data)))
    combined_data = train_data + test_data
    # 画分布图
    plt.figure(figsize=(10, 5))
    # weights = np.ones_like(combined_data) / len(combined_data)
    # plt.hist(combined_data, bins=100, label='Mixed Distribution', alpha=0.5, weights=weights)
    # weights = np.ones_like(train_data) / len(train_data)
    # plt.hist(train_data, bins=100, label='Train Distribution', alpha=0.5, weights=weights)
    # weights = np.ones_like(test_data) / len(test_data)
    # plt.hist(test_data, bins=100, label='Test Distribution', alpha=0.5, weights=weights)
    sns.kdeplot(combined_data, label='Mixed Distribution', alpha=0.5, shade=True)
    sns.kdeplot(train_data, label='Train Distribution', alpha=0.5, shade=True)
    sns.kdeplot(test_data, label='Test Distribution', alpha=0.5, shade=True)
    # 设置标题和轴标签
    plt.title(f'Data Distribution of mixed distribution at ratio {ratio} for {dataset_name} at {args.model_size} model')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dataset_name} {title} histogram at {args.model_size} model at ratio {ratio}.png')
    # 显示图表
    plt.show()
def remove_outliers(data, m=2):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)

    # 找到离群值
    outliers = data > mean + m * std

    # 计算没有离群值的平均值
    mean_without_outliers = np.mean(data[~outliers])

    # 用没有离群值的平均值替换离群值
    data[outliers] = mean_without_outliers

    return data.tolist()

def figure_draw(data_dict, title, args):
    plt.figure(figsize=(10, 5))
    fig, axs = plt.subplots(len(data_dict), figsize=(10, 5 * len(data_dict)))
    axs = np.atleast_2d(axs)
    for ax, (dataset_name, dataset_loss) in zip(axs.flatten(), data_dict.items()):
        for phase_name, phase_loss in dataset_loss.items():
            #weights = np.ones_like(phase_loss) / len(phase_loss)
            #ax.hist(phase_loss, bins=100, label=phase_name, alpha=0.5, weights=weights)
            sns.kdeplot(phase_loss, ax=ax, label=phase_name, alpha=0.5, bw_adjust=0.5, shade=True)
        ax.set_title(f'{dataset_name} {title} histogram  at {args.model_size} model')
        ax.set_xlabel(title)
        ax.set_ylabel('Percentage')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{title}_histograms_{args.model_size}_{args.dataset_name}.png")
    plt.show()


def min_prob_k(selected_log_probs):
    k_length = int(len(selected_log_probs) * 0.2)
    topk_log_prob = np.sort(selected_log_probs.cpu().numpy())[:k_length]
    min_k = -np.mean(topk_log_prob).item()
    return min_k

def min_prob_k_plus(probs, log_probs, selected_log_probs):
    #pdb.set_trace()
    mu = (probs * log_probs).to(torch.bfloat16).sum(-1)
    sigma = (probs.to(torch.bfloat16) * torch.square(log_probs.to(torch.bfloat16))).sum(-1) - torch.square(mu).to(torch.bfloat16)
    mink_plus = (selected_log_probs - mu) / (sigma.sqrt()+1e-9)
    k_length = int(len(mink_plus) * 0.2)
    topk = torch.sort(mink_plus.cpu())[:k_length]
    min_k_plus = -np.mean(topk).item()
    # if np.isnan(min_k_plus) or np.isinf(min_k_plus):
    #     pdb.set_trace()
    return min_k_plus

def caculate_outputs(model, tokenizer, text_batch):
    tokenized_inputs = tokenizer(text_batch,
                                 return_tensors="pt",
                                 truncation=True,
                                 padding=True,
                                 max_length=2048,
                                 )
    tokenized_inputs = {key: val.cuda(args.cuda) for key, val in tokenized_inputs.items()}
    target_labels = tokenized_inputs["input_ids"].clone()
    target_labels[tokenized_inputs["attention_mask"] == 0] = -100
    with torch.no_grad():
        outputs = model(**tokenized_inputs, labels=target_labels.cuda(args.cuda))
    return outputs, tokenized_inputs, target_labels

def caculate_loss_instance(idx, logits, target_labels):
    logits_i = logits[idx].unsqueeze(0)  # Shape (1, seq_length, vocab_size)
    target_i = target_labels[idx].unsqueeze(0)  # Shape (1, seq_length)
    shift_logits = logits_i[:, :-1, :].contiguous()
    shift_labels = target_i[:, 1:].contiguous()
    # 计算交叉熵损失并移除填充 token 贡献
    loss_i = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                             )
    # Create a mask to ignore the loss from padding tokens
    valid_mask = shift_labels != -100
    # 只有有效的 token 计算损失
    loss_i = loss_i * valid_mask.view(-1)
    # 计算每个样本的平均损失
    loss_i = loss_i.sum() / valid_mask.sum()
    return loss_i
def feature_collection(model, tokenizer, dataset, args, batch_size=8, upper_limit=10000, refer_model=None, refer_tokenizer=None):
    loss_collect = []
    mink_collect = []
    mink_plus_collect = []
    ppl_collect = []
    zlib_collect = []
    ref_loss_collect = []
    for batch in tqdm(batched_data(dataset, batch_size=batch_size)):
        batched_text = [item for item in batch]
        outputs,tokenized_inputs, target_labels = caculate_outputs(model, tokenizer, batched_text)
        if refer_model is not None:
            refer_outputs, refer_target_labels = caculate_outputs(refer_model, refer_tokenizer, batched_text)
        # tokenized_inputs = tokenizer(batched_text,
        #                              return_tensors="pt",
        #                              truncation=True,
        #                              padding=True,
        #                              max_length=2048,
        #                             )
        # tokenized_inputs = {key: val.cuda(args.cuda) for key, val in tokenized_inputs.items()}
        # target_labels = tokenized_inputs["input_ids"].clone()
        # target_labels[tokenized_inputs["attention_mask"] == 0] = -100
        # with torch.no_grad():
        #     outputs = model(**tokenized_inputs, labels=target_labels.cuda(args.cuda))
        #     # single_input_example = torch.tensor(tokenizer.encode(batch[0])).unsqueeze(0)
        #     # single_input_example = single_input_example.to(model.device)
        #     # single_output = model(single_input_example, labels=single_input_example)
        #     # single_loss, single_logits = single_output[:2]
        loss, logits = outputs[:2]
        log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        if refer_model is not None:
            ref_loss, ref_logits = refer_outputs[:2]
            ref_log_probabilities = torch.nn.functional.log_softmax(ref_logits, dim=-1)
            ref_probabilities = torch.nn.functional.softmax(ref_logits, dim=-1)


        # input_ids = single_input_example[0][1:].unsqueeze(-1)
        # probs = torch.nn.functional.softmax(single_logits[0, :-1], dim=-1)
        # log_probs = F.log_softmax(single_logits[0, :-1], dim=-1)
        # token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        # mu = (probs * log_probs).sum(-1)
        # sigma = (probs * torch.square(log_probs.to(torch.bfloat16))).sum(-1) - torch.square(mu)
        # mink_plus = (token_log_probs - mu) / sigma.sqrt()
        # 初始化
        all_prob = []
        # 获取每个样本的概率
        for idx in range(batch_size):
            # logits_i = logits[idx].unsqueeze(0)  # Shape (1, seq_length, vocab_size)
            # target_i = target_labels[idx].unsqueeze(0)  # Shape (1, seq_length)
            # shift_logits = logits_i[:, :-1, :].contiguous()
            # shift_labels = target_i[:, 1:].contiguous()
            # # 计算交叉熵损失并移除填充 token 贡献
            # loss_i = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
            #                          )
            # # Create a mask to ignore the loss from padding tokens
            # valid_mask = shift_labels != -100
            # # 只有有效的 token 计算损失
            # loss_i = loss_i * valid_mask.view(-1)
            # # 计算每个样本的平均损失
            # loss_i = loss_i.sum() / valid_mask.sum()
            loss_i = caculate_loss_instance(idx, logits, target_labels)
            if refer_model is not None:
                ref_loss_i = caculate_loss_instance(idx, ref_logits, refer_target_labels)
                ref_loss_collect.append(loss_i-ref_loss_i)
            input_ids_processed = tokenized_inputs["input_ids"][idx]
            attention_mask_processed = tokenized_inputs["attention_mask"][idx]
            log_probs = log_probabilities[idx]  # 形状为 (seq_length, vocab_size)
            probs = probabilities[idx]
            # 使用 attention_mask 筛选有效的 token
            valid_log_probs = log_probs[attention_mask_processed == 1]
            valid_token_ids = input_ids_processed[attention_mask_processed == 1]
            # 获取这些有效 token 的概率
            selected_log_probs = valid_log_probs.gather(-1, valid_token_ids.unsqueeze(1))
            #selected_probs = probs.gather(-1, valid_token_ids.unsqueeze(1))
            #selected_log_probs = valid_log_probs[np.arange(valid_token_ids.shape[0]), valid_token_ids]
            #selectd_probs = probs[np.arange(valid_token_ids.shape[0]), valid_token_ids]
            #pdb.set_trace()
            mink_plus = min_prob_k_plus(probs, log_probs, selected_log_probs)
            mink = min_prob_k(selected_log_probs)
            # 计算 topk 概率
            # k_length = int(len(selected_log_probs) * 0.2)
            # topk_log_prob = np.sort(selected_log_probs.cpu().numpy())[:k_length]
            # pred = -np.mean(topk_log_prob).item()
            # # perplexity's value
            ppl = torch.exp(loss).item()
            # 收集结果
            all_prob.append(selected_log_probs.cpu().numpy())
            mink_collect.append(mink)
            mink_plus_collect.append(mink_plus)
            ppl_collect.append(ppl)
            loss_collect.append(loss_i.item())
            zlib_collect.append(loss_i.cpu()/len(zlib.compress(bytes(batched_text[idx], "utf-8"))))
        if len(loss_collect) >= upper_limit:
            break
    loss_collect = remove_outliers(loss_collect)
    mink_collect = remove_outliers(mink_collect)
    ppl_collect = remove_outliers(ppl_collect)
    mink_plus_collect = remove_outliers(mink_plus_collect)
    zlib_collect = remove_outliers(zlib_collect)
    return loss_collect, mink_collect, ppl_collect, mink_plus_collect, zlib_collect, ref_loss_collect

def calculate_mean_var(dict, dataset_name):
    split_set = ["train", "valid", "test"]
    for idx1, set1 in enumerate(split_set):
        values = np.array(dict[dataset_name][set1])
        values = values[np.isnan(values)==False]
        mean = np.mean(values)
        var = np.var(values)
        std = np.std(values)
        kur = kurtosis(values)
        print("The mean, variance, std and kurtosis of {} in {} set are {},  {}, {} and {}".format(dataset_name, set1, mean, var, std, kur))
    return mean, var
def js_divergence(dict, dataset_name):
    # Ensure p and q sum to 1
    js_matrix = np.zeros((3, 3))
    split_set = ["train", "valid", "test"]
    for idx1, set1 in enumerate(split_set):
        for idx2, set2 in enumerate(split_set):
            values = np.array(dict[dataset_name][set1])
            values1 = values[np.isnan(values) == False]
            values = np.array(dict[dataset_name][set2])
            values2 = values[np.isnan(values) == False]
            hist1, bin_edges = np.histogram(values1, bins=300, density=True)
            hist2, _ = np.histogram(values2, bins=300, density=True)
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
            values = np.array(dict[dataset_name][set1])
            values1 = values[np.isnan(values) == False]
            values = np.array(dict[dataset_name][set2])
            values2 = values[np.isnan(values) == False]
            ks_stat, _ = ks_2samp(values1, values2)
            ks_matrix[idx1][idx2] = ks_stat
    return ks_matrix#close to zero means the two distributions are similar


#dataset_name = ["ArXiv", "DM Mathematics", "Enron Emails", "EuroParl", "FreeLaw", "Github", "Gutenberg (PG-19)",
#                "HackerNews", "NIH ExPorter", "PhilPapers", "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
#                "Ubuntu IRC", "USPTO Backgrounds", "Wikipedia (en)"]

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--model_size", type=str, default="160m")
parser.add_argument("--dataset_name", type=str, default="Pile-CC", choices=["ArXiv", "DM Mathematics", "Enron Emails",
                "EuroParl", "FreeLaw", "Github", "Gutenberg (PG-19)", "HackerNews", "NIH ExPorter", "PhilPapers",
                "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange","Ubuntu IRC",
                "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA"])
parser.add_argument("--cuda", type=int, default=0, help="cuda device")
parser.add_argument("--skip_calculation", type=str, default="True")
parser.add_argument("--reference_model", type=str, default="True")
parser.add_argument("--samples", type=int, default=5000)
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
    if args.reference_model == "True":
        refer_model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-base-alpha-3b")
        refer_tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-3b")
    else:
        refer_model = None
        refer_tokenizer = None
    tokenizer.pad_token = tokenizer.eos_token
    loss_dict = {}
    prob_dict = {}
    ppl_dict = {}
    mink_plus_dict = {}
    zlib_dict = {}
    refer_dict = {}
    loss_dict[args.dataset_name] = {"train": [], "valid": [], "test": []}
    prob_dict[args.dataset_name] = {"train": [], "valid": [], "test": []}
    ppl_dict[args.dataset_name] = {"train": [], "valid": [], "test": []}
    mink_plus_dict[args.dataset_name] = {"train": [], "valid": [], "test": []}
    zlib_dict[args.dataset_name] = {"train": [], "valid": [], "test": []}
    refer_dict[args.dataset_name] = {"train": [], "valid": [], "test": []}
    if args.dataset_name == "WikiMIA":
        for text_len in [32, 64, 128, 256]:
            dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{text_len}")
            member_data = dataset.filter(lambda example: example['label'] == 1)
            non_member_data = dataset.filter(lambda example: example['label'] == 0)
            if text_len == 32:
                mia_dataset = DatasetDict({
                    'train': member_data["input"],
                    'test': non_member_data["input"],
                    'valid': non_member_data["input"]
                })
            else:
                mia_dataset["train"].extend(member_data["input"])
                mia_dataset["test"].extend(non_member_data["input"])
                mia_dataset["valid"].extend(non_member_data["input"])
    for split in ["train", "valid", "test"]:
        if split in ["test", "valid"]:
            if args.dataset_name == "WikiMIA":
                dataset = mia_dataset[split]
            else:
                dataset = torch.load(f"by_dataset/{split}_{args.dataset_name}.pt")
        else:
            if args.dataset_name == "WikiMIA":
                dataset = mia_dataset[split]
            else:
                for i in range(1):
                    dataset = torch.load(f"by_dataset/{split}_{args.dataset_name}_{i}.pt")
        loss_list, prob_list, ppl_list, mink_plus_list, zlib_list, refer_list = feature_collection(model, tokenizer, dataset, args,
                                                                                       batch_size=args.batch_size,
                                                                                       upper_limit=args.samples,
                                                                                       refer_model=refer_model,
                                                                                       refer_tokenizer=refer_tokenizer)
        loss_dict[args.dataset_name][split].extend(loss_list)
        prob_dict[args.dataset_name][split].extend(prob_list)
        ppl_dict[args.dataset_name][split].extend(ppl_list)
        mink_plus_dict[args.dataset_name][split].extend(mink_plus_list)
        zlib_dict[args.dataset_name][split].extend(zlib_list)
        refer_dict[args.dataset_name][split].extend(refer_list)
    pickle.dump(loss_dict, open(f"feature_result/{args.dataset_name}_{args.model_size}_loss_dict.pkl", "wb"))
    pickle.dump(prob_dict, open(f"feature_result/{args.dataset_name}_{args.model_size}_prob_dict.pkl", "wb"))
    pickle.dump(ppl_dict, open(f"feature_result/{args.dataset_name}_{args.model_size}_ppl_dict.pkl", "wb"))
    pickle.dump(mink_plus_dict, open(f"feature_result/{args.dataset_name}_{args.model_size}_mink_plus_dict.pkl", "wb"))
    pickle.dump(zlib_dict, open(f"feature_result/{args.dataset_name}_{args.model_size}_zlib_dict.pkl", "wb"))
    pickle.dump(refer_dict, open(f"feature_result/{args.dataset_name}_{args.model_size}_refer_dict.pkl", "wb"))
loss_dict = pickle.load(open(f"feature_result/{args.dataset_name}_{args.model_size}_loss_dict.pkl", "rb"))
prob_dict = pickle.load(open(f"feature_result/{args.dataset_name}_{args.model_size}_prob_dict.pkl", "rb"))
ppl_dict = pickle.load(open(f"feature_result/{args.dataset_name}_{args.model_size}_ppl_dict.pkl", "rb"))
mink_plus_dict = pickle.load(open(f"feature_result/{args.dataset_name}_{args.model_size}_mink_plus_dict.pkl", "rb"))
zlib_dict = pickle.load(open(f"feature_result/{args.dataset_name}_{args.model_size}_zlib_dict.pkl", "rb"))
figure_draw(loss_dict, "Loss", args)
figure_draw(prob_dict, "Prob", args)
figure_draw(ppl_dict, "PPL", args)
figure_draw(mink_plus_dict, "Mink_plus", args)
figure_draw(zlib_dict, "Zlib", args)
mix_distribution(loss_dict, args.dataset_name, "Loss", args)
mix_distribution(prob_dict, args.dataset_name, "Prob", args)
mix_distribution(ppl_dict, args.dataset_name, "PPL", args)
mix_distribution(mink_plus_dict, args.dataset_name, "Mink_plus", args)
mix_distribution(zlib_dict, args.dataset_name, "Zlib", args)
f = open(f"results/{args.dataset_name}_{args.model_size}_results.txt", "w")
for idx, dict in enumerate([loss_dict, prob_dict, ppl_dict, mink_plus_dict, zlib_dict]):
    if idx == 0:
        print("Loss Distribution Similarity Matrix")
        f.write("Loss Distribution Similarity Matrix\n")
    elif idx == 1:
        print("Prob Distribution Similarity Matrix")
        f.write("Prob Distribution Similarity Matrix\n")
    elif idx == 2:
        print("PPL Distribution Similarity Matrix")
    elif idx == 3:
        f.write("Mink_plus Distribution Similarity Matrix\n")
        print("Mink_plus Distribution Similarity Matrix")
    else:
        print("Zlib Distribution Similarity Matrix")
        f.write("Zlib Distribution Similarity Matrix\n")
    calculate_mean_var(dict, args.dataset_name)
    js_matrix = js_divergence(dict, args.dataset_name)
    print(js_matrix)
    f.write(str(js_matrix) + '\n')
    ks_matrix = ks_hypothesis(dict, args.dataset_name)
    print(ks_matrix)
    f.write(str(ks_matrix) + '\n')
f.close()


