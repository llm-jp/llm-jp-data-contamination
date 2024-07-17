import json
import pandas as pd
from eval import create_random_samples
import torch
from collections import defaultdict
from sklearn.metrics import auc, roc_curve
import zlib
from torch.nn import CrossEntropyLoss
from scipy.stats import entropy, ks_2samp, kurtosis, wasserstein_distance
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict
from itertools import islice

def batched_data(dataset, batch_size):
    data_iter = iter(dataset)
    while True:
        batch = list(islice(data_iter, batch_size))
        if not batch:
            break
        yield batch

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

def calculate_mink_and_mink_plus(batch_logits, batched_tokenized_inputs):
    batch_input_ids = batched_tokenized_inputs["input_ids"][:, 1:].unsqueeze(-1)
    target_labels = batched_tokenized_inputs["input_ids"].clone()
    target_labels[batched_tokenized_inputs["attention_mask"] == 0] = -100
    batch_probs = F.softmax(batch_logits[:, :-1].float(), dim=-1)
    batch_log_probs = F.log_softmax(batch_logits[:, :-1].float(), dim=-1)
    mask = target_labels[:, 1:] != -100
    mask = mask.unsqueeze(-1)
    batch_token_log_probs = batch_log_probs.gather(dim=-1, index=batch_input_ids).squeeze(-1)
    batch_probs_masked = batch_probs.where(mask, 0)
    batch_log_probs_masked = batch_log_probs.where(mask, 0)
    batch_mu = (batch_probs_masked.float() * batch_log_probs_masked.float()).float().sum(-1)
    batch_sigma =  ((batch_probs_masked.float() * torch.square(torch.where(batch_probs_masked > 0,batch_log_probs_masked.float(),  torch.tensor(0.0, device=batch_log_probs_masked.device, dtype=torch.float32)))).sum(dim=-1)- torch.square(batch_mu.float()).squeeze())
    mask = mask.squeeze(-1)
    batch_mink_plus = (batch_token_log_probs - batch_mu).float() * mask / batch_sigma.float().sqrt()
    token_length = mask.sum(dim=1)
    batch_mink_plus[mask == False] = torch.inf
    batch_token_log_probs[mask == False] = torch.inf
    sorted_mink_plus, _ = torch.sort(batch_mink_plus)
    sorted_mink, _ = torch.sort(batch_token_log_probs)
    batch_mink_plus_avg = []
    batch_mink_avg = []
    for i, length in enumerate(token_length):
        front_values = sorted_mink_plus[i, :length]
        avg = torch.mean(front_values.float()).item()
        batch_mink_plus_avg.append(avg)
        if torch.tensor(avg) == torch.inf:
            pdb.set_trace()
        front_values = sorted_mink[i, :length]
        avg = torch.mean(front_values.float()).item()
        batch_mink_avg.append(avg)
    return batch_mink_plus_avg, batch_mink_avg


def caculate_instance_loss_perplexity_zlib(batch_logits, target_labels, batched_text):
    shift_logits = batch_logits[:, :-1, :].contiguous()
    labels = target_labels[:, 1:].contiguous()
    loss_fct = CrossEntropyLoss(reduction='none')
    pdb.set_trace()
    lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
    instance_losses = lm_loss.view(-1, shift_logits.size(1))
    loss_value_list = []
    ppl_value_list = []
    zlib_value_list = []
    for idx, i in enumerate(instance_losses):
        loss = i.sum() / sum(i != 0)
        loss_value_list.append(loss.item())
        ppl = torch.exp(loss.float()).item()
        ppl_value_list.append(ppl)
        zlib_value = loss.float().cpu() / (len(zlib.compress(bytes(batched_text[idx], "utf-8")))+1)
        zlib_value_list.append(zlib_value.item())
    return loss_value_list, ppl_value_list, zlib_value_list

def remove_outliers(data, m=2):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    # 找到大于均值 + m * std 和小于均值 - m * std 的离群值
    outliers_high = data > mean + m * std
    outliers_low = data < mean - m * std
    outliers = outliers_high | outliers_low
    # 计算没有离群值的平均值
    mean_without_outliers = np.mean(data[~outliers])
    # 用没有离群值的平均值替换离群值
    data[outliers] = mean_without_outliers
    return data.tolist()


def save_jsonl(data, fpath):
    with open(fpath, 'w+', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
def load_json(fpath):
    with open(fpath) as f:
        return json.load(f)

def clean_text(text):
    words_list = []
    text = text.lower().replace("sentence 2:", '').replace(".", '')
    for sentence in text.split(','):
        words = sentence.replace('["', '')\
                .replace('\\n', '')\
                .replace('"]', '')\
                .replace("['", '')\
                .replace("]", '').split(' ')
        words_list += words
    if '' in words_list: 
        for _ in range(words_list.count('')):
            words_list.remove('')
    return words_list

def process_data_to_json(dataset_name, data):
    if dataset_name == "databricks-dolly-15k-ja":
        data['input'] = (
                '### 指示：\n' + data['instruction'] + '\n\n' +
                '### 文脈 :\n' + data['context'].replace('', '（空）')
        )
        data['input'] = data['input'].str.replace('### 文脈 :\n（空）', '')
        data['output'] = '### 応答:\n' + data['response']
        result = data[['input', 'output']]
        result = result.to_dict(orient='records')
        return result


def load_data(dataset_name, split_name, num_samples):
    if dataset_name in ["alt-e-to-j", "alt-j-to-e","chabsa", "jamp", "janli",
                          "jcommonsenseqa", "jemhopqa", "jmmlu", "jnli", "jsem",
                          "jsick", "jsquad","jsts", "mawps", "niilc"]:
        loaded_data = load_json(f"datasets_contamination/1.3.0/evaluation/{split_name}/{dataset_name}.json")
        random_samples = create_random_samples(loaded_data["samples"], num_samples=num_samples)
    elif dataset_name == "databricks-dolly-15k-ja":
        df = pd.read_json("hf://datasets/llm-jp/databricks-dolly-15k-ja/databricks-dolly-15k-ja.jsonl", lines=True)
        loaded_data = process_data_to_json("databricks-dolly-15k-ja", df)
        random_samples = create_random_samples(loaded_data, num_samples=num_samples)
    elif dataset_name == "oasst1-21k-ja":
        df = pd.read_json("hf://datasets/llm-jp/oasst1-21k-ja/oasst1-21k-ja.jsonl", lines=True)
    elif dataset_name == "databricks-dolly-15k-ja":
        df = pd.read_json("hf://datasets/llm-jp/oasst1-21k-en/oasst1-21k-en.jsonl", lines=True)
        random_samples = df.sample(num_samples)
    return random_samples

def obtain_instruction_time_travel(dataset_name, split_name, model_name=None):
    if model_name == "llm-jp-v2":
        chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '\\n\\n### 指示:\\n' + message['content'] }}{% elif message['role'] == 'system' %}{{ '\\n\\n### 指示:\\n' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\\n\\n### 応答:\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\\n\\n### 応答:\\n' }}{% endif %}{% endfor %}"
    else:
        chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{message['content'] }}{% elif message['role'] == 'system' %}{{message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\\n### 回答:\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\\n### 回答:\\n' }}{% endif %}{% endfor %}"
    if dataset_name in ["jnli", "jsick", "jamp","janli"]:
        guided_chat = [
            {"role": "system",
             "content": f"次の文1は、{dataset_name}データセットの{split_name}分割から提供されています。\n文1の後にあるラベルは、文1と文2の間の論理的な関係を示します。\n文2はデータセット内のインスタンスと完全に一致する必要があります。\nデータセットに表示された通りに、文2を完成させてください。必ず文2のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "以下のラベルが文1と文2の論理的関係を示すように、文1を基に文2を完成させる。\n必ず文2のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["alt-e-to-j", "alt-j-to-e"]:
        if dataset_name == "alt-e-to-j":
            info = "英語から日本語"
        else:
            info = "日本語から英語"
        guided_chat = [
            {"role": "system",
             "content": f"次の文は、{dataset_name}データセットの{split_name}分割から提供されています。\n原文の前半と原文全体の{info}への翻訳文は提供されています。\nデータセットに表示された通りに、原文の後半を完成させてください。必ずその原文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": f"原文の前半と原文全体の{info}への翻訳文は提供されています。必ずその原文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jemhopqa"]:
        guided_chat = [
                    {"role": "system",
                     "content": f"次の質問文の前半と答えは、{dataset_name}データセットの{split_name}分割から提供されています。\n質問文の下の半分を出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、質問文の下の半分を出力してください。それ以外には何も含めないことを厳守してください。"},
                    {"role": "user", "content": ""},
                ]
        general_chat = [
            {"role": "system",
             "content": f"質問文の半分と答えは提供されいます。\n質問文の下の半分を出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、質問文の下の半分を出力してください。それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jmmlu"]:
        guided_chat = [
                    {"role": "system",
                     "content": f"次の質問文と答えは、{dataset_name}データセットの{split_name}分割から提供されています。\nその質問文と答えを参考し、ABCDという四つの選択肢を出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、選択肢を出力してください。必ずABCDの選択肢のみを出力し、それ以外には何も含めないことを厳守してください。"},
                    {"role": "user", "content": ""},
                ]
        general_chat = [
            {"role": "system",
             "content": f"質問文と答えは提供されています。\nその質問文と答えを参考し、ABCDという四つの選択肢を出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、選択肢を出力してください。必ずABCDの選択肢のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jcommonsenseqa"]:
        guided_chat = [
            {"role": "system",
             "content": f"次の質問文と答えは、{dataset_name}データセットの{split_name}分割から提供されています。\nその質問文と答えを参考し、この答えと質問文に合う01234という五つの選択肢を順番に出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、選択肢を出力してください。必ず01234の選択肢のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": f"質問文と答えは提供されています。\nその質問文と答えを参考し、この答えと質問文に合う01234という五つの選択肢を順番に出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、選択肢を出力してください。必ず01234の選択肢のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jsts"]:
        guided_chat = [
            {"role": "system",
             "content": f"次の文1は、{dataset_name}データセットの{split_name}分割から提供されています。\n文1の後にある数字は、文1と文2の間の類似度を示します。0.0に近いほど文ペアの意味が異なり、5.0に近いほど文ペアの意味が似ていることを表しています。\n文2はデータセット内のインスタンスと完全に一致する必要があります。\nデータセットに表示された通りに、文2を完成させてください。文1とその類似度を使って、必ず文2のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "文1の後にある数字は、文1と文2の間の類似度を示します。0.0に近いほど文ペアの意味が異なり、5.0に近いほど文ペアの意味が似ていることを表しています。\n文1とその類似度を使って、必ず文2のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["niilc"]:
        guided_chat = [
            {"role": "system",
             "content": f"次の質問に対する答えと質問文の前半は、{dataset_name}データセットの{split_name}分割から提供されています。\nデータセットに表示された通りに、質問文の後半を完成させてください。\nその文はデータセット内のインスタンスと完全に一致する必要があります。必ず質問文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "質問に対する答えと質問文の前半は提供されています。\n必ず質問文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["mawps"]:
        guided_chat = [
            {"role": "system",
             "content": f"次の計算問題に対する答えと質問文の前半は、{dataset_name}データセットの{split_name}分割から提供されています。\データセットに表示された通りに、質問文の後半を完成させてください。必ずその質問文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "計算問題に対する答えと質問文の前半は提供されています。\n必ずその質問文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jsquad"]:
        guided_chat = [
            {"role": "system",
             "content": f"文章と文章に対する答えは、{dataset_name}データセットの{split_name}分割から提供されています。\nデータセットに表示された通りに、その文章と答えに合う質問文を書いてください。必ず質問文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "文章と文章に対する答えは提供されています。\nその文章と答えに合う質問文を書いてください。必ず質問文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jsem"]:
        guided_chat = [
            {"role": "system",
             "content": f"前提と、前提と仮説の関係の答えは、{dataset_name}データセットの{split_name}分割から提供されています。\nその答えはyes、no、unknown、undefの中からの答えと提供されています\nデータセットに表示された通りに、仮説文を完成させてください。必ず仮説文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "前提と、前提と仮説の関係の答えはyes、no、unknown、undefから提供されています\nデータセットに表示された通りに、仮説文を書いてください。必ず仮説文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["chabsa"]:
        guided_chat = [
            {"role": "system",
             "content": f"与えられた文章の前半と、全体の文章から抽出された固有表現で書かれたターゲットの名前と、それぞれの名前に対するpositive、neutral、negativeの極性は、{dataset_name}データセットの{split_name}分割から提供されています。\nデータセットに表示された通りに、未完成の文章の後半を書いてください。必ず文章の後半の文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "与えられた文章の前半と、全体の文章から抽出された固有表現で書かれたターゲットの名前と、それぞれの名前に対するpositive、neutral、negativeの極性は提供されています\n未完成の文章の後半を書いてください。必ず文章の後半の文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name == "databricks-dolly-15k-ja":
        if model_name == "llm-jp-v2":
            chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '' + message['content'] }}{% elif message['role'] == 'system' %}{{ '' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '' }}{% endif %}{% endfor %}"
        else:
            chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{message['content'] }}{% elif message['role'] == 'system' %}{{message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '' }}{% endif %}{% endfor %}"
        guided_chat = [
            {"role": "system",
             "content": "最初の指示文と文脈文と応答の一部が提供されます。それらの情報に基づいて、データセット中の応答文の原形の残った文を完成させなさい。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat,general_chat, chat_template
def obtain_instruction_naive(dataset_name, split_name, model_name=None):
    if model_name == "llm-jp-v2":
        chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '' + message['content'] }}{% elif message['role'] == 'system' %}{{ '' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '' }}{% endif %}{% endfor %}"
    else:
        chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{message['content'] }}{% elif message['role'] == 'system' %}{{message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '' }}{% endif %}{% endfor %}"
    if dataset_name == "databricks-dolly-15k-ja":
        general_chat = [
            {"role": "system",
             "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"},
            {"role": "user", "content": ""},
        ]
    return general_chat, chat_template


def formalize_input_time_travel(dataset_name,guided_chat, general_chat, inst_type, example):
    if dataset_name in ["jnli", "jsick", "jamp", "janli"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        procesesd_sent1 = example['input'].split('\n')[0].replace('前提：', '')
        sent1 = f"文1: {procesesd_sent1}"
        sent2 = "文2: "+example['input'].split("\n")[1].replace("仮説：", "")
        label = "含意" if example['output'] == "entailment" else "矛盾" if example['output'] == "contradiction" else "中立"
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\nラベル:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\nラベル:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["alt-e-to-j", "alt-j-to-e"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        if dataset_name == "alt-e-to-j":
            sent1 = " ".join(example['input'].split(" ")[:len(example['input'].split(" ")) // 2])
            sent1 = sent1.strip()
            label = example['output']
            sent2 = " ".join(example['input'].split(" ")[len(example['input'].split(" "))//2:])
            sent2= sent2.strip()
        else:
            sent1 = example['input'][:len(example['input']) // 2]
            sent1 = sent1.strip()
            label = example['output']
            sent2 = example['input'][len(example['input']) // 2:]
            sent2 = sent2.strip()
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"原文:{sent1}\n翻訳文:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"原文:{sent1}\n翻訳文:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["jemhopqa", "niilc", "mawps"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1 = example['input'][:len(example['input'])//2]
        label = example['output']
        sent2 = example['input'][len(example['input'])//2:]
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\n答え:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\n答え:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["jsts"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1, sent2 = example['input'].split('\n')
        label = example['output']
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1.strip()}\n類似度:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1.strip()}\n類似度:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["jcommonsenseqa", "jmmlu"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1 = example['input'].split("\n")[0]
        label = example['output']
        sent2 = example['input'].split("\n")[1]
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\n答え:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\n答え:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["jsquad"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1, sent2 = example['input'].split('\n')
        label = example['output']
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\n答え:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\n答え:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["jsem"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1, sent2 = example['input'].split('\n')
        label = example['output']
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\n前提と仮説の関係の答え:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\n前提と仮説の関係の答え:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["chabsa"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1, sent2 = example['input'][:len(example['input'])//2], example['input'][len(example['input'])//2:]
        label = example['output']
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\nターゲットの名前とそれぞれの極性:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\nターゲットの名前とそれぞれの極性:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name == "databricks-dolly-15k-ja":
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1 = example['input']
        prev_half = example["output"][:len(example["output"])//2]
        sent2 = example["output"][len(example["output"])//2:]
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\n"+prev_half
        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\n"+prev_half
        return chat, f"{sent1}\n"+prev_half, sent2, instruction

def formalize_input_baseline(dataset_name, general_chat, example):
    if dataset_name == "databricks-dolly-15k-ja":
        sent1 = example['input']
        prev_half = example["output"][:len(example["output"])//2]
        sent2 = example["output"][len(example["output"])//2:]
        chat = general_chat
        chat[1]["content"] = f"{sent1}\n"+prev_half
        return chat, f"{sent1}\n"+prev_half, sent2, chat[0]["content"]+f"{sent1}\n"+prev_half
    

def calculate_perplexity(output, tokenized_input):
    # 提取生成的logits和生成的序列
    logits = torch.stack(output.scores, dim=1)[0]  # (sequence_length, vocab_size)
    generated_seq = output.sequences[0]  # (sequence_length_with_prompt + generated_tokens)
    # 提取生成部分的目标序列
    input_length = tokenized_input.size(-1)
    target_seq = generated_seq[input_length:]
    # 确保logits和目标序列对齐
    assert logits.size(0) == target_seq.size(0), "Logits and target sequence length must match."
    # 计算交叉熵损失和困惑度
    perplexities = []
    with torch.no_grad():
        for i in range(logits.size(0)):  # iterate over time steps
            # 获取当前 step 的 logits 和 label
            current_logits = logits[i, :].unsqueeze(0)  # (1, vocab_size)
            current_label = target_seq[i].unsqueeze(0)  # (1,)
            # 计算交叉熵损失
            loss = F.cross_entropy(current_logits, current_label, reduction='none')
            # 计算困惑度
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
    return perplexities

def calculate_memorization_score(output, tokenized_input, continuation, tokenizer):
    # 提取生成的序列
    generated_seq = output.sequences[0]  # (sequence_length_with_prompt + generated_tokens)
    # 提取生成部分的目标序列
    input_length = tokenized_input.size(-1)
    target_seq = generated_seq[input_length:]
    # 计算生成部分的准确率
    correct = 0
    for i in range(min(len(continuation), target_seq.size(0))):
        if target_seq[i] == tokenizer.eos_token_id:
            break  # 如果遇到eos_token，停止计数
        if target_seq[i] == continuation[i]:
            correct += 1
    return correct / target_seq.size(0) if target_seq.size(0) > 0 else 0
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def do_plot(prediction, answers, sweep_fn=sweep, metric='auc', legend="", output_dir=None):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """
    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr<.05)[0][-1]]
    # bp()
    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@5%%FPR of %.4f\n'%(legend, auc,acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc

    plt.plot(fpr, tpr, label=legend+metric_text)
    return legend, auc,acc, low

def fig_fpr_tpr(all_output, output_dir):
    print("output_dir", output_dir)
    answers = []
    metric2predictions = defaultdict(list)
    for ex in all_output:
        answers.append(ex["label"])
        for metric in ex["pred"].keys():
            if ("raw" in metric) and ("clf" not in metric):
                continue
            metric2predictions[metric].append(ex["pred"][metric])

    plt.figure(figsize=(4, 3))
    with open(f"{output_dir}/auc.txt", "w") as f:
        for metric, predictions in metric2predictions.items():
            legend, auc, acc, low = do_plot(predictions, answers, legend=metric, metric='auc', output_dir=output_dir)
            f.write('%s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f\n' % (legend, auc, acc, low))

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig(f"{output_dir}/auc.png")
