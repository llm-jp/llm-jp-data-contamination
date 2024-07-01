from datasets import load_dataset
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
import pickle
from itertools import islice
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pdb


def batched_data(dataset, batch_size):
    data_iter = iter(dataset)
    while True:
        batch = list(islice(data_iter, batch_size))
        if not batch:
            break
        yield batch

def figure_draw(data_dict, title):
    plt.figure(figsize=(10, 5))
    fig, axs = plt.subplots(len(data_dict), figsize=(10, 5 * len(data_dict)))
    axs = np.atleast_2d(axs)
    for ax, (dataset_name, dataset_loss) in zip(axs.flatten(), data_dict.items()):
        for phase_name, phase_loss in dataset_loss.items():
            weights = np.ones_like(phase_loss) / len(phase_loss)
            ax.hist(phase_loss, bins=50, label=phase_name, alpha=0.5, weights=weights)
        ax.set_title(f'{title} values histogram for {dataset_name}')
        ax.set_xlabel(title)
        ax.set_ylabel('Percentage')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{title}_histograms.png")
    plt.show()



def loss_collection(model, dataset, batch_size=8):
    loss_collect = []
    prob_collect = []
    ppl_collect = []
    for batch in tqdm(batched_data(dataset, batch_size=batch_size)):
        tokenized_inputs = tokenizer([item for item in batch],
                                     return_tensors="pt",
                                     truncation=True,
                                     padding=True,
                                     max_length=2048)
        tokenized_inputs = {key: val.to("cuda") for key, val in tokenized_inputs.items()}
        with torch.no_grad():
            outputs = model(**tokenized_inputs, labels=tokenized_inputs["input_ids"].cuda())
        loss, logits = outputs[:2]
        probabilities = torch.nn.functional.log_softmax(logits, dim=2)
        loss_collect.append(loss.item())
        mask = tokenized_inputs["attention_mask"].bool()
        # Get loss for each token where mask is 1
        pdb.set_trace()
        all_probs = torch.masked_select(probabilities, mask.unsqueeze(-1)).reshape(mask.sum(dim=-1), -1)
        # Number of top values to take for each set of probabilities
        k_lengths = (mask.sum(dim=-1) * 0.2).int()
        # Find the lowest `k_length` values for each set of probabilities
        topk_probs, _ = all_probs.topk(k_lengths, dim=-1, largest=False)
        # Calculate the mean of the top k probabilites
        preds = -(topk_probs.mean(dim=-1)).tolist()
        # Add to list
        prob_collect.extend(preds)
        # Compute the perplexity for all samples in the batch
        ppl = torch.exp(loss).tolist()
        # Add to list
        ppl_collect.extend(ppl)
        pdb.set_trace()
    return loss_collect, prob_collect, ppl_collect

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
loss_dict = {}
prob_dict = {}
ppl_dict = {}
for name in dataset_name:
    loss_dict[name] = {"train": [], "valid": [], "test": []}
    prob_dict[name] = {"train": [], "valid": [], "test": []}
    ppl_dict[name] = {"train": [], "valid": [], "test": []}
    for split in split_name:
        if split in ["test", "valid"]:
            dataset = torch.load(f"by_dataset/{split}_{name}.pt")
            loss_list, prob_list, ppl_list = loss_collection(model, dataset)
            loss_dict[name][split].extend(loss_list)
            prob_dict[name][split].extend(prob_list)
            ppl_dict[name][split].extend(ppl_list)
        else:
            for i in range(1):
                dataset = torch.load(f"by_dataset/{split}_{name}_{i}.pt")
                loss_list, prob_list, ppl_list = loss_collection(model, dataset)
                loss_dict[name][split].extend(loss_list)
                prob_dict[name][split].extend(prob_list)
                ppl_dict[name][split].extend(ppl_list)
pickle.dump(loss_dict, open("loss_dict.pkl", "wb"))
pickle.dump(prob_dict, open("prob_dict.pkl", "wb"))
pickle.dump(ppl_dict, open("ppl_dict.pkl", "wb"))
figure_draw(loss_dict, "Loss")
figure_draw(prob_dict, "Prob")
figure_draw(ppl_dict, "PPL")


