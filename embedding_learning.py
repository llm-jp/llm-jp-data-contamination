import os
import pdb
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from utils import form_dataset, batched_data_with_indices, clean_dataset
import argparse
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

layer_results = pd.DataFrame(columns=["Dataset", "Layer", "Train Accuracy", "Test Accuracy"])
layer_results_path = "layer_results.csv"
def pad_embeddings(embed_list, attn_mask_list, max_length):
    padded_embed_list = []
    attention_masks = []
    for embed, attn_mask in zip(embed_list, attn_mask_list):
        padding_size = max_length - embed.shape[1]
        if padding_size > 0:
            pad = torch.nn.functional.pad(embed, (0, 0, 0, padding_size), value=0)  # ensure padding value is 0
            attention_mask = torch.nn.functional.pad(attn_mask, (0, padding_size), value=0)  # pad attention_mask as well
        else:
            pad = embed
            attention_mask = attn_mask
        padded_embed_list.append(pad)
        attention_masks.append(attention_mask)
    return torch.cat(padded_embed_list, dim=0), torch.cat(attention_masks, dim=0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout=0.1, max_len=5000):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, attention_mask):
        # x: Batch_size x Seq_len x Input_dim
        x = self.embedding(x)
        x = self.positional_encoding(x)

        attention_mask_inv = ~attention_mask.bool()

        x = self.layer_norm(x)
        x = self.transformer(x, src_key_padding_mask=attention_mask_inv)
        x = self.layer_norm(x)

        mask = attention_mask.unsqueeze(-1).expand_as(x)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--model_size", type=str, default="6.9b")
parser.add_argument("--dataset_name", type=str, default="Pile-CC", choices=["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                     "pubmed_central", "wikipedia_(en)", "full_pile", "all"])
parser.add_argument("--cuda", type=int, default=1, help="cuda device")
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--prepare_dataset", type=str, default="True")
args = parser.parse_args()

if args.dataset_name == "all":
    dataset_names = ["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                     "pubmed_central", "wikipedia_(en)", "full_pile","WikiMIA32", "WikiMIA64", "WikiMIA128","WikiMIA256",
                      "WikiMIAall"]
else:
    dataset_names = [args.dataset_name]

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
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.generation_config.output_hidden_states = True
#model.generation_config.output_attentions = True
model.generation_config.output_scores = True
model.generation_config.return_dict_in_generate = True

layer_num = model.config.num_hidden_layers

for dataset_name in dataset_names:
    if "WikiMIA" in dataset_name:
        dataset = form_dataset(dataset_name)
        dataset["member"] = dataset["train"]
        dataset["nonmember"] = dataset["test"]
    else:
        dataset = load_dataset("iamgroot42/mimir", dataset_name,
                               split="ngram_13_0.2") if dataset_name != "full_pile" else load_dataset(
            "iamgroot42/mimir",
            "full_pile",
            split="none")
    device = f'cuda:{args.cuda}'
    member_embed_list = []
    non_member_embed_list = []
    member_attn_mask_list = []
    nonmember_attn_mask_list = []
    for layer_index in tqdm(range(layer_num)):
        for set_name in ["member", "nonmember"]:
            cleaned_data, orig_indices = clean_dataset(dataset[set_name], dataset_name, online=True)
            for idx, (data_batch, orig_indices_batch) in tqdm(enumerate(batched_data_with_indices(cleaned_data, orig_indices, batch_size=args.batch_size))):
                if idx * args.batch_size > args.samples:
                    break
                batched_text = [item for item in data_batch]
                tokenized_inputs = tokenizer(batched_text,
                                             return_tensors="pt",
                                             truncation=True,
                                             padding=True,
                                             max_length=2048,
                                             )
                tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}
                target_labels = tokenized_inputs["input_ids"].clone().to(device)
                target_labels[tokenized_inputs["attention_mask"] == 0] = -100
                with torch.no_grad():
                    outputs = model(**tokenized_inputs, labels=target_labels, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states
                context_embedding = hidden_states[-2]
                layer_num = len(hidden_states)
                if set_name == "member":
                    member_embed_list.append(context_embedding.cpu())
                    member_attn_mask_list.append(tokenized_inputs["attention_mask"].cpu())
                elif set_name == "nonmember":
                    non_member_embed_list.append(context_embedding.cpu())
                    nonmember_attn_mask_list.append(tokenized_inputs["attention_mask"].cpu())
        max_length = max(max(embed.shape[1] for embed in member_embed_list),
                         max(embed.shape[1] for embed in non_member_embed_list))

        # 填充并合并embedding
        member_embeddings, member_attn_masks = pad_embeddings(member_embed_list, member_attn_mask_list, max_length)
        nonmember_embeddings, nonmember_attn_masks = pad_embeddings(non_member_embed_list, nonmember_attn_mask_list, max_length)

        # 创建标签
        member_labels = torch.ones(member_embeddings.shape[0])
        nonmember_labels = torch.zeros(nonmember_embeddings.shape[0])

        # 合并数据和标签
        X = torch.cat([member_embeddings, nonmember_embeddings], axis=0)
        y = torch.cat([member_labels, nonmember_labels], axis=0)
        attention_masks = torch.cat([member_attn_masks, nonmember_attn_masks], dim=0)

        # 将数据分为训练集和测试集
        X_train, X_test, y_train, y_test, attn_train, attn_test = train_test_split(X, y, attention_masks, test_size=0.2, random_state=42)

        # 转换数据为tensor
        X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, member_embeddings.shape[1], member_embeddings.shape[2])
        X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, member_embeddings.shape[1], member_embeddings.shape[2])
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        attn_train = torch.tensor(attn_train, dtype=torch.float32)
        attn_test = torch.tensor(attn_test, dtype=torch.float32)

        batch_size = 4  # 可根据需要调整
        train_dataset = TensorDataset(X_train, y_train, attn_train)
        test_dataset = TensorDataset(X_test, y_test, attn_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_dim = member_embeddings.shape[2] # 输入的特征维度
        hidden_dim = 256
        output_dim = 2
        num_layers = 4
        num_heads = 8

        pred_model = TransformerClassifier(input_dim, hidden_dim, output_dim, num_layers, num_heads)


        # 使用GPU
        device = f'cuda:{args.cuda}'
        pred_model = pred_model.to(device)
        X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

        # 定义损失和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(pred_model.parameters(), lr=0.0001)
        # 训练模型
        num_epochs = 10
        for epoch in range(num_epochs):
            pred_model.train()
            for i, (inputs, labels, attention_masks) in enumerate(train_loader):
                inputs, labels, attention_masks = inputs.to(device), labels.to(device), attention_masks.to(device)
                optimizer.zero_grad()
                outputs = pred_model(inputs, attention_masks)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if (i + 1) % 10 == 0:  # 每10个批次打印一次loss
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            pred_model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels, attention_masks in train_loader:
                    inputs, labels, attention_masks = inputs.to(device), labels.to(device), attention_masks.to(device)
                    outputs = pred_model(inputs, attention_masks)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            train_accuracy = accuracy_score(all_labels, all_preds)
            print(f'Train Accuracy of {dataset_name} at Layer {layer_index} at Epoch {epoch}: {train_accuracy:.4f}')
            print(classification_report(all_labels, all_preds, target_names=['Nonmember', 'Member']))


        #Test Accuracy: 0.5450 model size 2.8 arxiv
        #Test Accuracy: 0.5341 model size 2.8 hackernews
        # 评估模型
        pred_model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels, attention_masks in test_loader:
                inputs, labels, attention_masks = inputs.to(device), labels.to(device), attention_masks.to(device)
                outputs = pred_model(inputs, attention_masks)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        test_accuracy = accuracy_score(all_labels, all_preds)
        print(f'Test Accuracy of {dataset_name} at Layer {layer_index}: {test_accuracy:.4f}')
        print(classification_report(all_labels, all_preds, target_names=['Nonmember', 'Member']))
        layer_results = layer_results._append({
            "Dataset": dataset_name,
            "Layer": layer_index,
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy},
            ignore_index=True)
os.makedirs("embedding_learning", exist_ok=True)
csv_file_path = f"embedding_learning/{args.model_size}/learning_results.csv"
if os.path.exists(csv_file_path):
    layer_results.to_csv(f"embedding_learning/{args.model_size}/learning_results.csv", mode='a', header=False, index=False)
else:
    layer_results.to_csv(f"embedding_learning/{args.model_size}/learning_results.csv", index=False)


