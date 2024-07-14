from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

def min_prob_k(selected_log_probs):
    k_length = int(len(selected_log_probs) * 1)
    topk_log_prob = np.sort(selected_log_probs.cpu().numpy())[:k_length]
    min_k = -np.mean(topk_log_prob).item()
    return min_k

def min_prob_k_plus(probs, log_probs, selected_log_probs):
    #pdb.set_trace()
    mu = (probs * log_probs).to(torch.bfloat16).sum(-1)
    sigma = (probs.to(torch.bfloat16) * torch.square(log_probs.to(torch.bfloat16))).sum(-1) - torch.square(mu).to(torch.bfloat16)
    mink_plus = (selected_log_probs - mu) / (sigma.sqrt())
    k_length = int(len(mink_plus) * 1)
    topk, _ = torch.sort(mink_plus.cpu())[:k_length]
    min_k_plus = -np.mean(topk, dim=1).item()
    # if np.isnan(min_k_plus) or np.isinf(min_k_plus):
    #     pdb.set_trace()
    return min_k_plus


def calculate_mink_and_mink_plus(batch_logits, batched_tokenized_inputs):
    batch_input_ids = batched_tokenized_inputs["input_ids"][:, 1:].unsqueeze(-1)
    batch_probs = F.softmax(batch_logits[:, :-1].to(device), dim=-1)
    batch_log_probs = F.log_softmax(batch_logits[:, :-1].to(device), dim=-1)
    mask = target_labels[:, 1:] != -100
    mask = mask.unsqueeze(-1)
    batch_token_log_probs = batch_log_probs.gather(dim=-1, index=batch_input_ids.to(device)).squeeze(-1)
    batch_probs_masked = batch_probs.where(mask, 0)
    batch_log_probs_masked = batch_log_probs.where(mask, 0)
    batch_mu = (batch_probs_masked * batch_log_probs_masked).to(torch.bfloat16).sum(-1)
    batch_sigma = (batch_probs_masked * torch.square(batch_log_probs_masked.to(torch.bfloat16))).sum(
        dim=-1) - torch.square(batch_mu)
    mask = mask.squeeze()
    batch_mink_plus = (batch_token_log_probs - batch_mu).to(torch.bfloat16) * mask / batch_sigma.sqrt()
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
        front_values = sorted_mink[i, :length]
        avg = torch.mean(front_values.float()).item()
        batch_mink_avg.append(avg)
    return batch_mink_plus_avg, batch_mink_avg

model = GPTNeoXForCausalLM.from_pretrained(
      f"EleutherAI/pythia-160m-deduped",
      revision="step143000",
      cache_dir=f"./pythia-70m-deduped/step143000",
    ).eval()
#model = model.to_bettertransformer()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(
  f"EleutherAI/pythia-160m-deduped",
  revision="step143000",
  cache_dir=f"./pythia-70m-deduped/step143000",
)
tokenizer.pad_token = tokenizer.eos_token
for instance_text in ["I love you", "Me hate you and love him"]:
    inputs = tokenizer(instance_text, return_tensors="pt", padding=True)
    tokenized_inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**tokenized_inputs, labels=tokenized_inputs["input_ids"])
    loss, logits = outputs[:2]
    print(loss)
    ll = -loss.item() # log-likelihood
    input_ids = inputs["input_ids"][0][1:].unsqueeze(-1)
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids.to(device)).squeeze(-1)
    mu = (probs * log_probs).to(torch.bfloat16).sum(-1)
    sigma = (probs * torch.square(log_probs.to(torch.bfloat16))).sum(-1) - torch.square(mu)
    mink_plus = (token_log_probs - mu) / sigma.sqrt()#

batch_text = ["I love you", "Me hate you and love him"]
batch_inputs = tokenizer(batch_text, return_tensors="pt", padding=True, max_length=6)
batched_tokenized_inputs = {key: val.to(device) for key, val in batch_inputs.items()}
target_labels = batched_tokenized_inputs["input_ids"].clone()
target_labels[batched_tokenized_inputs["attention_mask"] == 0] = -100
with torch.no_grad():
     outputs = model(**batched_tokenized_inputs, labels=target_labels.to(device))
batch_loss, batch_logits = outputs[:2]

# we are doing next-token prediction; shift prediction scores and input ids by one
shift_logits = batch_logits[:, :-1, :].contiguous()
labels = target_labels[:, 1:].contiguous()
loss_fct = CrossEntropyLoss(reduction='none')
lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
instance_losses = lm_loss.view(-1, shift_logits.size(1))
for i in instance_losses:
    print(i.sum()/sum(i!=0))

# batch_ll = -batch_loss.item() # log-likelihood
batch_input_ids = batched_tokenized_inputs["input_ids"][:, 1:].unsqueeze(-1)
batch_probs = F.softmax(batch_logits[:, :-1].to(device), dim=-1)
batch_log_probs = F.log_softmax(batch_logits[:, :-1].to(device), dim=-1)
#mask = torch.ones_like(batch_probs, dtype=torch.bool)
mask = target_labels[:, 1:] != -100
mask = mask.unsqueeze(-1)
batch_token_log_probs = batch_log_probs.gather(dim=-1, index=batch_input_ids.to(device)).squeeze(-1)
batch_probs_masked = batch_probs.where(mask, 0)  # 对 batch_probs 应用 mask
batch_log_probs_masked = batch_log_probs.where(mask, 0)
batch_mu = (batch_probs_masked * batch_log_probs_masked).to(torch.bfloat16).sum(-1)
batch_sigma = (batch_probs_masked * torch.square(batch_log_probs_masked.to(torch.bfloat16))).sum(dim=-1) - torch.square(batch_mu)
mask = mask.squeeze()
batch_mink_plus = (batch_token_log_probs - batch_mu).to(torch.bfloat16)*mask / batch_sigma.sqrt()
token_length = mask.sum(dim=1)
batch_mink_plus[mask==False]=torch.inf
batch_token_log_probs[mask==False]=torch.inf
sorted_mink_plus, _ = torch.sort(batch_mink_plus)
sorted_mink, _ = torch.sort(batch_token_log_probs)
batch_mink_plus_avg = []
batch_mink_avg = []
for i, length in enumerate(token_length):
    front_values = sorted_mink_plus[i, :length]  # 选择前 length 个元素
    avg = torch.mean(front_values.float()).item()  # 计算平均值，确保为 float 类型以返回正确的平均值
    batch_mink_plus_avg.append(avg)
    front_values = sorted_mink[i, :length]
    avg = torch.mean(front_values.float()).item()
    batch_mink_avg.append(avg)
print(batch_mink_plus_avg)
print(batch_mink_avg)
