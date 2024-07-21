from datasets import load_dataset
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch.nn.functional as F
import argparse
import random
import seaborn as sns
from datasets import DatasetDict
import os
from torch.nn import CrossEntropyLoss
from utils import *


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
        ax.set_ylabel('Density')
        ax.legend()
    plt.tight_layout()
    os.makedirs(f"figures/{dataset_name}", exist_ok=True)
    plt.savefig(f"figures/{dataset_name}/{title}_histograms_{args.model_size}_{dataset_name}.png")
    plt.show()


def caculate_outputs(model, tokenizer, text_batch, device):
    tokenized_inputs = tokenizer(text_batch,
                                 return_tensors="pt",
                                 truncation=True,
                                 padding=True,
                                 max_length=2048,
                                 )
    tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
    target_labels = tokenized_inputs["input_ids"].clone().to(device)
    target_labels[tokenized_inputs["attention_mask"] == 0] = -100
    with torch.no_grad():
        outputs = model(**tokenized_inputs, labels=target_labels)
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
    device = f'cuda:{args.cuda}'
    loss_collect = []
    mink_collect = []
    mink_plus_collect = []
    ppl_collect = []
    zlib_collect = []
    ref_loss_collect = []
    idx_list = []
    cleaned_data, orig_indices = clean_dataset(dataset)
    for idx, (data_batch, orig_indices_batch) in tqdm(
            enumerate(batched_data_with_indices(cleaned_data, orig_indices, batch_size=args.batch_size))):
        orig_idx = [item for item in orig_indices_batch]
        batched_text = [item for item in data_batch]
        outputs,tokenized_inputs, target_labels = caculate_outputs(model, tokenizer, batched_text, device=device)
        if refer_model is not None:
            refer_outputs, refer_tokenized_inputs, refer_target_labels = caculate_outputs(refer_model, refer_tokenizer, batched_text, device=device)
        batch_mink_plus_avg, batch_mink_avg = calculate_mink_and_mink_plus(outputs[1], tokenized_inputs)
        loss_value_list, ppl_value_list, zlib_value_list = caculate_instance_loss_perplexity_zlib(outputs[1], target_labels, batched_text)
        mink_plus_collect.extend(batch_mink_plus_avg)
        mink_collect.extend(batch_mink_avg)
        loss_collect.extend(loss_value_list)
        ppl_collect.extend(ppl_value_list)
        zlib_collect.extend(zlib_value_list)
        idx_list.extend(orig_idx)
        if refer_model is not None:
            ref_loss, ref_logits = refer_outputs[:2]
            ref_log_probabilities = torch.nn.functional.log_softmax(ref_logits, dim=-1)
            ref_probabilities = torch.nn.functional.softmax(ref_logits, dim=-1)
        if len(loss_collect) >= upper_limit:
            break
        ref_loss_collect.extend(ref_loss)
    # loss_collect = remove_outliers(loss_collect)
    # mink_collect = remove_outliers(mink_collect)
    # ppl_collect = remove_outliers(ppl_collect)
    # mink_plus_collect = remove_outliers(mink_plus_collect)
    # zlib_collect = remove_outliers(zlib_collect)
    return loss_collect, mink_collect, ppl_collect, mink_plus_collect, zlib_collect, ref_loss_collect, idx_list

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



def results_caculate_and_draw(dataset_name, args):
    loss_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_loss_dict.pkl", "rb"))
    prob_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_prob_dict.pkl", "rb"))
    ppl_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_ppl_dict.pkl", "rb"))
    mink_plus_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_mink_plus_dict.pkl", "rb"))
    zlib_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_zlib_dict.pkl", "rb"))
    refer_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_refer_dict.pkl", "rb"))
    idx_list = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_idx_list.pkl", "rb"))
    all_dict = [loss_dict, prob_dict, ppl_dict, mink_plus_dict, zlib_dict, refer_dict]
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
        elif idx == 4:
            figure_draw(zlib_dict, "Zlib", dataset_name, args)
            mix_distribution(zlib_dict, dataset_name, "Zlib", args)
            print("Zlib Distribution Similarity Matrix")
            f.write("Zlib Distribution Similarity Matrix\n")
        elif idx == 5:
            residual_dict = {}
            residual_dict[dataset_name]=  {"train": [], "valid": [], "test": []}
            for split in ["train", "valid", "test"]:
                residual_dict[split] = [loss_dict[dataset_name][split][i] - refer_dict[dataset_name][split][i]
                                        for i in range(len(loss_dict[dataset_name][split]))]
            figure_draw(residual_dict, "Refer", dataset_name, args)
            mix_distribution(residual_dict, dataset_name, "Refer", args)
            print("Refer Distribution Similarity Matrix")
            f.write("Refer Distribution Similarity Matrix\n")
        print(idx)
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
parser.add_argument("--gradient_collection", type=str, default=False)
args = parser.parse_args()

if args.dataset_name == "all":
    dataset_names = ["ArXiv", "DM Mathematics",
                  "FreeLaw", "Github",  "HackerNews", "NIH ExPorter",
                 "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                 "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA"]
    #dataset_names = ["PubMed Central", "StackExchange",
    #                "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA"]
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
      torch_dtype=torch.bfloat16,
    ).cuda(args.cuda).eval()
    #model = model.to_bettertransformer()
    tokenizer = AutoTokenizer.from_pretrained(
      f"EleutherAI/pythia-{args.model_size}-deduped",
      revision="step143000",
      cache_dir=f"./pythia-{args.model_size}-deduped/step143000",
    )
    if args.reference_model == "True":
        refer_model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-base-alpha-3b-v2",
                                                           trust_remote_code=True,
                                                           torch_dtype="auto").cuda(args.cuda).eval()
        refer_tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-3b-v2")
    else:
        refer_model = None
        refer_tokenizer = None
    tokenizer.pad_token = tokenizer.eos_token
    refer_tokenizer.pad_token = refer_tokenizer.eos_token
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
            loss_list, prob_list, ppl_list, mink_plus_list, zlib_list, refer_list, idx_list = feature_collection(model, tokenizer, dataset[split], args,
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
        pickle.dump(idx_list, open(f"feature_result/{dataset_name}_{args.model_size}_idx_list.pkl", "wb"))
        results_caculate_and_draw(dataset_name, args)




