from datasets import load_dataset
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
import pickle
from itertools import islice
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def batched_data(dataset, batch_size):
    data_iter = iter(dataset)
    while True:
        batch = list(islice(data_iter, batch_size))
        if not batch:
            break
        yield batch


def loss_collection(model, dataset):
    loss_list = []
    for batch in tqdm(batched_data(dataset, batch_size=8)):
        tokenized_inputs = tokenizer([item for item in batch],
                                     return_tensors="pt",
                                     truncation=True,
                                     padding=True,
                                     max_length=2048)
        tokenized_inputs = {key: val.to("cuda") for key, val in tokenized_inputs.items()}
        outputs = model(**tokenized_inputs, labels=tokenized_inputs["input_ids"].cuda())
        loss = outputs.loss
        loss_list.append(loss.item())
    return loss_list

#dataset_name = ["ArXiv", "DM Mathematics", "Enron Emails", "EuroParl", "FreeLaw", "Github", "Gutenberg (PG-19)",
#                "HackerNews", "NIH ExPorter", "PhilPapers", "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
#                "Ubuntu IRC", "USPTO Backgrounds", "Wikipedia (en)"]
dataset_name = ["Pile-CC"]
split_name = ["train", "valid", "test"]

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step143000",
  cache_dir="./pythia-160m-deduped/step143000",
).half().eval()
model = model.to_bettertransformer()
model = model.cuda()
tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step143000",
  cache_dir="./pythia-160m-deduped/step143000",
)
tokenizer.pad_token = tokenizer.eos_token
loss_dicit = {}
for name in dataset_name:
    loss_dicit[name] = {"train": [], "valid": [], "test": []}
    for split in split_name:
        if split in ["test", "valid"]:
            dataset = torch.load(f"by_dataset/{split}_{name}.pt")
            loss_list = loss_collection(model, dataset)
            loss_dicit[name][split].extend(loss_list)
        else:
            for i in range(1):
                dataset = torch.load(f"by_dataset/{split}_{name}_{i}.pt")
                loss_list = loss_collection(model, dataset)
                loss_dicit[name][split].extend(loss_list)
pickle.dump(loss_dicit, open("loss_dict.pkl", "wb"))
plt.figure(figsize=(10, 5))
fig, axs = plt.subplots(len(loss_dicit), figsize=(10, 5 * len(loss_dicit)))
axs = np.atleast_2d(axs)
for ax, (dataset_name, dataset_loss) in zip(axs.flatten(), loss_dicit.items()):
    for phase_name, phase_loss in dataset_loss.items():
        weights = np.ones_like(phase_loss) / len(phase_loss)
        ax.hist(phase_loss, bins=50, label=phase_name, alpha=0.5, weights=weights)
    ax.set_title(f'Loss values histogram for {dataset_name}')
    ax.set_xlabel('Loss')
    ax.set_ylabel('Percentage')
    ax.legend()
plt.tight_layout()
plt.savefig("loss_histograms.png")
plt.show()

