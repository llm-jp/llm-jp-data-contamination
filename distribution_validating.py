from datasets import load_dataset
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
import pickle
from itertools import islice

def batched_data(dataset, batch_size):
    data_iter = iter(dataset)
    while True:
        batch = list(islice(data_iter, batch_size))
        if not batch:
            break
        yield batch


def loss_collection(model, dataset):
    loss_list = []
    for batch in batched_data(dataset, batch_size=32):
        tokenized_inputs = tokenizer([item["text"] for item in batch],
                                     return_tensors="pt",
                                     truncation=True,
                                     padding=True,
                                     max_length=2048)
        outputs = model(**tokenized_inputs, labels=tokenized_inputs["input_ids"])
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
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step143000",
  cache_dir="./pythia-160m-deduped/step143000",
)
loss_dicit = {}
for name in dataset_name:
    loss_dicit[name] = {"train": [], "valid": [], "test": []}
    for split in split_name:
        if split in ["test", "valid"]:
            dataset = torch.load(f"/model/pile/by_dataset/{split}_name.pt")
            loss_list = loss_collection(model, dataset)
            loss_dicit[name][split].extend(loss_list)
        else:
            for i in range(5):
                dataset = torch.load(f"/model/pile/by_dataset/{split}_name_{i}.pt")
                loss_list = loss_collection(model, dataset)
                loss_dicit[name][split].extend(loss_list)
pickle.dump(loss_dicit, open("loss_dict.pkl", "wb"))


