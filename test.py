from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
import torch
import torch.nn.functional as F
model = GPTNeoXForCausalLM.from_pretrained(
      f"EleutherAI/pythia-160m-deduped",
      revision="step143000",
      cache_dir=f"./pythia-160m-deduped/step143000",
    ).half().eval()
model = model.to_bettertransformer()
model = model.cuda(1)
tokenizer = AutoTokenizer.from_pretrained(
  f"EleutherAI/pythia-160m-deduped",
  revision="step143000",
  cache_dir=f"./pythia-160m-deduped/step143000",
)
tokenizer.pad_token = tokenizer.eos_token

instance_text = ["I love you"]
inputs = tokenizer(instance_text, return_tensors="pt", padding=True)
tokenized_inputs = {key: val.cuda(1) for key, val in inputs.items()}
with torch.no_grad():
    outputs = model(**tokenized_inputs, labels=tokenized_inputs["input_ids"])
loss, logits = outputs[:2]
ll = -loss.item() # log-likelihood
input_ids = inputs["input_ids"][0][1:].unsqueeze(-1)
probs = F.softmax(logits[0, :-1], dim=-1)
log_probs = F.log_softmax(logits[0, :-1], dim=-1)
token_log_probs = log_probs.gather(dim=-1, index=input_ids.cuda(1)).squeeze(-1)
mu = (probs * log_probs).to(torch.bfloat16).sum(-1).sum(-1)
sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

batch_text = ["I love you", "I hate you and love him"]
batch_inputs = tokenizer(instance_text, return_tensors="pt", padding=True)
batched_tokenized_inputs = {key: val.cuda(1) for key, val in batch_inputs.items()}
with torch.no_grad():
    outputs = model(**batched_tokenized_inputs)
batch_loss, batch_logits = outputs[:2]
batch_ll = -loss.item() # log-likelihood
batch_input_ids = batch_inputs["input_ids"][0][1:].unsqueeze(-1)
batch_probs = F.softmax(batch_logits[0, :-1], dim=-1)
batch_log_probs = F.log_softmax(batch_logits[0, :-1], dim=-1)
token_log_probs = batch_log_probs.gather(dim=-1, index=batch_input_ids).squeeze(-1)
batch_mu = (batch_probs * batch_log_probs).to(torch.bfloat16).sum(-1).sum(-1)
batch_sigma = (batch_probs * torch.square(batch_log_probs.to(torch.bfloat16).sum(-1))).sum(-1) - torch.square(batch_mu)

