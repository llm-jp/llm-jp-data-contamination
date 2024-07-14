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
    sns.kdeplot(combined_data, label='Mixed Distribution', alpha=1, shade=True)
    sns.kdeplot(train_data, label='Train Distribution', alpha=1, shade=True)
    sns.kdeplot(test_data, label='Test Distribution', alpha=1, shade=True)
    # 设置标题和轴标签
    plt.title(f'Data Distribution of mixed distribution at ratio {ratio} for {dataset_name} at {args.model_size} model')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    os.makedirs(f"mixed_figure/{dataset_name}", exist_ok=True)
    plt.savefig(f'mixed_figure/{dataset_name}/{title} histogram at {args.model_size} model at ratio {ratio}.png')
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

def figure_draw(data_dict, title,dataset_name, args):
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
    os.makedirs(f"figures/{dataset_name}", exist_ok=True)
    plt.savefig(f"figures/{dataset_name}/{title}_histograms_{args.model_size}_{dataset_name}.png")
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
    try:
        logits_i = logits[idx].unsqueeze(0)  # Shape (1, seq_length, vocab_size)
    except:
        pdb.set_trace()
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
        loss, logits = outputs[:2]
        log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        if refer_model is not None:
            ref_loss, ref_logits = refer_outputs[:2]
            ref_log_probabilities = torch.nn.functional.log_softmax(ref_logits, dim=-1)
            ref_probabilities = torch.nn.functional.softmax(ref_logits, dim=-1)
        # 初始化
        all_prob = []
        # 获取每个样本的概率
        for idx in range(logits.shape[0]):
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
            mink_plus = min_prob_k_plus(probs, log_probs, selected_log_probs)
            mink = min_prob_k(selected_log_probs)
            # 计算 topk 概率
            # # perplexity's value
            ppl = torch.exp(loss).item()
            # 收集结果
            all_prob.append(selected_log_probs.cpu().numpy())
            mink_collect.append(mink)
            mink_plus_collect.append(mink_plus)
            ppl_collect.append(ppl)
            loss_collect.append(loss_i.item())
            zlib_collect.append(loss_i.cpu()/len(zlib.compress(bytes(batched_text[idx], "utf-8"))))
        #pdb.set_trace()
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
    ks_statistic_matrix = np.zeros((3, 3))
    ks_p_value_matrix = np.zeros((3, 3))
    split_set = ["train", "valid", "test"]
    for idx1, set1 in enumerate(split_set):
        for idx2, set2 in enumerate(split_set):
            values = np.array(dict[dataset_name][set1])
            values1 = values[np.isnan(values) == False]
            values = np.array(dict[dataset_name][set2])
            values2 = values[np.isnan(values) == False]
            ks_stat, p_value = ks_2samp(values1, values2)
            ks_statistic_matrix[idx1][idx2] = ks_stat
            ks_p_value_matrix[idx1][idx2] = p_value
    return ks_statistic_matrix, ks_p_value_matrix#close to zero means the two distributions are similar

def wasserstein_distance_caculate(dict, dataset_name):
    ws_matrix = np.zeros((3, 3))
    split_set = ["train", "valid", "test"]
    for idx1, set1 in enumerate(split_set):
        for idx2, set2 in enumerate(split_set):
            values = np.array(dict[dataset_name][set1])
            values1 = values[np.isnan(values) == False]
            values = np.array(dict[dataset_name][set2])
            values2 = values[np.isnan(values) == False]
            ws_stat = wasserstein_distance(values1, values2)
            ws_matrix[idx1][idx2] = ws_stat
    return ws_matrix#close to zero means the two distributions are similar

def form_dataset(dataset_name):
    if dataset_name == "WikiMIA":
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
        return mia_dataset
    else:
        train_dataset = torch.load(f"/model/pile/by_dataset/train_{dataset_name}_0.pt")
        valid_dataset = torch.load(f"/model/pile/by_dataset/valid_{dataset_name}.pt")
        test_dataset = torch.load(f"/model/pile/by_dataset/test_{dataset_name}.pt")
        dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset,
            'valid': valid_dataset
        })
        return dataset



def results_caculate_and_draw(dataset_name, args):
    loss_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_loss_dict.pkl", "rb"))
    prob_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_prob_dict.pkl", "rb"))
    ppl_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_ppl_dict.pkl", "rb"))
    mink_plus_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_mink_plus_dict.pkl", "rb"))
    zlib_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_zlib_dict.pkl", "rb"))
    all_dict = [loss_dict, prob_dict, ppl_dict, mink_plus_dict, zlib_dict]
    f = open(f"results/{dataset_name}_{args.model_size}_results.txt", "w")
    for idx, dict in enumerate(all_dict):
        if idx == 0:
            figure_draw(loss_dict, "Loss", dataset_name, args)
            mix_distribution(loss_dict, dataset_name, "Loss", args)
            print("Loss Distribution Similarity Matrix")
            f.write("Loss Distribution Similarity Matrix\n")
        elif idx == 1:
            figure_draw(prob_dict, "Prob", dataset_name, args)
            mix_distribution(prob_dict, dataset_name, "Prob", args)
            print("Prob Distribution Similarity Matrix")
            f.write("Prob Distribution Similarity Matrix\n")
        elif idx == 2:
            figure_draw(ppl_dict, "PPL", dataset_name, args)
            mix_distribution(ppl_dict, dataset_name, "PPL", args)
            print("PPL Distribution Similarity Matrix")
            f.write("PPL Distribution Similarity Matrix\n")
        elif idx == 3:
            figure_draw(mink_plus_dict, "Mink_plus", dataset_name, args)
            mix_distribution(mink_plus_dict, dataset_name, "Mink_plus", args)
            f.write("Mink_plus Distribution Similarity Matrix\n")
            print("Mink_plus Distribution Similarity Matrix")
        else:
            figure_draw(zlib_dict, "Zlib", dataset_name, args)
            mix_distribution(zlib_dict, dataset_name, "Zlib", args)
            print("Zlib Distribution Similarity Matrix")
            f.write("Zlib Distribution Similarity Matrix\n")
        calculate_mean_var(dict, dataset_name)
        js_matrix = js_divergence(dict, dataset_name)
        print(js_matrix)
        f.write(str(js_matrix) + '\n')
        ks_matrix, ks_p_value_matrix = ks_hypothesis(dict, dataset_name)
        print(ks_matrix)
        f.write(str(ks_matrix) + '\n')
        print(ks_p_value_matrix)
        f.write(str(ks_p_value_matrix) + '\n')
        ws_matrix = wasserstein_distance_caculate(dict, dataset_name)
        print(ws_matrix)
        f.write(str(ws_matrix) + '\n')
    f.close()


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--model_size", type=str, default="160m")
parser.add_argument("--dataset_name", type=str, default="Pile-CC", choices=["ArXiv", "DM Mathematics",
                 "FreeLaw", "Github",  "HackerNews", "NIH ExPorter",
                "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA", "all"])
parser.add_argument("--cuda", type=int, default=0, help="cuda device")
parser.add_argument("--skip_calculation", type=str, default="True")
parser.add_argument("--reference_model", type=str, default="True")
parser.add_argument("--samples", type=int, default=5000)
args = parser.parse_args()

if args.dataset_name == "all":
    #dataset_names = ["ArXiv", "DM Mathematics",
    #                 "FreeLaw", "Github", "HackerNews", "NIH ExPorter",
    #                  "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
    #                  "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA"]
    dataset_names = ["Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                    "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA"]
else:
    dataset_names = [args.dataset_name]

if args.skip_calculation == "True":
    skip_calculation = True
    for dataset_name in dataset_names:
        results_caculate_and_draw(dataset_name, args)
else:
    skip_calculation = False
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
    for dataset_name in dataset_names:
        dataset = form_dataset(dataset_name)
        loss_dict = {}
        prob_dict = {}
        ppl_dict = {}
        mink_plus_dict = {}
        zlib_dict = {}
        refer_dict = {}
        loss_dict[dataset_name] = {"train": [], "valid": [], "test": []}
        prob_dict[dataset_name] = {"train": [], "valid": [], "test": []}
        ppl_dict[dataset_name] = {"train": [], "valid": [], "test": []}
        mink_plus_dict[dataset_name] = {"train": [], "valid": [], "test": []}
        zlib_dict[dataset_name] = {"train": [], "valid": [], "test": []}
        refer_dict[dataset_name] = {"train": [], "valid": [], "test": []}
        for split in ["train", "valid", "test"]:
            loss_list, prob_list, ppl_list, mink_plus_list, zlib_list, refer_list = feature_collection(model, tokenizer, dataset[split], args,
                                                                                           batch_size=args.batch_size,
                                                                                           upper_limit=args.samples,
                                                                                           refer_model=refer_model,
                                                                                           refer_tokenizer=refer_tokenizer)
            loss_dict[dataset_name][split].extend(loss_list)
            prob_dict[dataset_name][split].extend(prob_list)
            ppl_dict[dataset_name][split].extend(ppl_list)
            mink_plus_dict[dataset_name][split].extend(mink_plus_list)
            zlib_dict[dataset_name][split].extend(zlib_list)
            refer_dict[dataset_name][split].extend(refer_list)
        pickle.dump(loss_dict, open(f"feature_result/{dataset_name}_{args.model_size}_loss_dict.pkl", "wb"))
        pickle.dump(prob_dict, open(f"feature_result/{dataset_name}_{args.model_size}_prob_dict.pkl", "wb"))
        pickle.dump(ppl_dict, open(f"feature_result/{dataset_name}_{args.model_size}_ppl_dict.pkl", "wb"))
        pickle.dump(mink_plus_dict, open(f"feature_result/{dataset_name}_{args.model_size}_mink_plus_dict.pkl", "wb"))
        pickle.dump(zlib_dict, open(f"feature_result/{dataset_name}_{args.model_size}_zlib_dict.pkl", "wb"))
        pickle.dump(refer_dict, open(f"feature_result/{dataset_name}_{args.model_size}_refer_dict.pkl", "wb"))
        results_caculate_and_draw(dataset_name, args)




