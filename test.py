from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
import torch
import torch.nn.functional as F
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

instance_text = ["I love you"]
inputs = tokenizer(instance_text, return_tensors="pt", padding=True)
tokenized_inputs = {key: val.to(device) for key, val in inputs.items()}
with torch.no_grad():
    outputs = model(**tokenized_inputs, labels=tokenized_inputs["input_ids"])
loss, logits = outputs[:2]
ll = -loss.item() # log-likelihood
input_ids = inputs["input_ids"][0][1:].unsqueeze(-1)
probs = F.softmax(logits[0, :-1], dim=-1)
log_probs = F.log_softmax(logits[0, :-1], dim=-1)
token_log_probs = log_probs.gather(dim=-1, index=input_ids.to(device)).squeeze(-1)
mu = (probs * log_probs).to(torch.bfloat16).sum(-1).sum(-1)
sigma = (probs * torch.square(log_probs.to(torch.bfloat16))).sum(-1) - torch.square(mu)
#
batch_text = ["I love you", "Me hate you and love him"]
batch_inputs = tokenizer(batch_text, return_tensors="pt", padding=True)
batched_tokenized_inputs = {key: val.to(device) for key, val in batch_inputs.items()}
target_labels = batched_tokenized_inputs["input_ids"].clone()
target_labels[batched_tokenized_inputs["attention_mask"] == 0] = -100
with torch.no_grad():
     outputs = model(**batched_tokenized_inputs, labels=target_labels.to(device))
batch_loss, batch_logits = outputs[:2]
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
batch_mu = (batch_probs_masked * batch_log_probs_masked).to(torch.bfloat16).sum(-1).sum(-1)
batch_sigma = (batch_probs_masked * torch.square(batch_log_probs_masked.to(torch.bfloat16))).sum(-1) - torch.square(batch_mu).unsqueeze(-1)
mask = mask.squeeze()
token_length = mask.sum(dim=1)
batch_sigma[mask==False]=torch.inf
sorted_value, _ = torch.sort(batch_sigma)
averages = []
for i, length in enumerate(token_length):
    front_values = sorted_value[i, :length]  # 选择前 length 个元素
    avg = torch.mean(front_values.float()).item()  # 计算平均值，确保为 float 类型以返回正确的平均值
    averages.append(avg)
