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

def pad_embeddings(embed_list, max_length):
    padded_embed_list = []
    for embed in embed_list:
        padding_size = max_length - embed.shape[1]
        if padding_size > 0:
            pad = torch.nn.functional.pad(embed, (0, 0, 0, padding_size))
        else:
            pad = embed
        padded_embed_list.append(pad)
    return torch.cat(padded_embed_list, dim=0)



parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--model_size", type=str, default="2.8b")
parser.add_argument("--dataset_name", type=str, default="Pile-CC", choices=["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                     "pubmed_central", "wikipedia_(en)", "full_pile", "all"])
parser.add_argument("--cuda", type=int, default=1, help="cuda device")
parser.add_argument("--samples", type=int, default=100)
parser.add_argument("--prepare_dataset", type=str, default="True")
args = parser.parse_args()

if args.dataset_name == "all":
    dataset_names = ["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                     "pubmed_central", "wikipedia_(en)", "full_pile","WikiMIA64", "WikiMIA128","WikiMIA256",
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
            if set_name == "member":
                member_embed_list.append(context_embedding.cpu())
            elif set_name == "nonmember":
                non_member_embed_list.append(context_embedding.cpu())

max_length = max(max(embed.shape[1] for embed in member_embed_list),
                 max(embed.shape[1] for embed in non_member_embed_list))

# 填充并合并embedding
member_embeddings = pad_embeddings(member_embed_list, max_length)
nonmember_embeddings = pad_embeddings(non_member_embed_list, max_length)

# 创建标签
member_labels = torch.ones(member_embeddings.shape[0])
nonmember_labels = torch.zeros(nonmember_embeddings.shape[0])

# 合并数据和标签
X = torch.cat([member_embeddings, nonmember_embeddings], axis=0)
y = torch.cat([member_labels, nonmember_labels], axis=0)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1))

# 转换数据为tensor
X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, member_embeddings.shape[1], member_embeddings.shape[2])
X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, member_embeddings.shape[1], member_embeddings.shape[2])
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# 定义二元分类模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads):
        super(TransformerClassifier, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=1)  # 对时间维度进行池化操作
        x = self.fc(x)
        return x


# 模型的超参数
input_dim = member_embeddings.shape[2]
hidden_dim = 256  # 可以根据需要调整
output_dim = 2
num_layers = 2
num_heads = 4

model = TransformerClassifier(input_dim, hidden_dim, output_dim, num_layers, num_heads)

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

# 定义损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
    print(f'Test Accuracy: {accuracy:.4f}')
    print(classification_report(y_test.cpu(), predicted.cpu(), target_names=['Nonmember', 'Member']))


