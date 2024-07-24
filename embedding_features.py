import pdb
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from utils import form_dataset, batched_data_with_indices, clean_dataset
import argparse
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import roc_auc_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--model_size", type=str, default="1b")
parser.add_argument("--dataset_name", type=str, default="all", choices=["ArXiv", "DM Mathematics",
                 "FreeLaw", "Github",  "HackerNews", "NIH ExPorter",
                "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA", "all"])
parser.add_argument("--cuda", type=int, default=1, help="cuda device")
parser.add_argument("--skip_calculation", type=str, default="True")
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--gradient_collection", type=str, default=False)
args = parser.parse_args()

if args.dataset_name == "all":
    dataset_names =["ArXiv", "DM Mathematics",
                  "FreeLaw", "Github",  "HackerNews", "NIH ExPorter",
                 "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                 "USPTO Backgrounds", "Wikipedia (en)","WikiMIA32","WikiMIA64", "WikiMIA128","WikiMIA256",
                     "WikiMIAall"]
    #dataset_names = ["Github", "HackerNews", "NIH ExPorter","Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
    #                "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA"]
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

results_df = pd.DataFrame(
    columns=['Dataset Name', 'Layer Index', 'DB Index',
             'Silhouette Score', 'Calinski Harabasz Index'])
for dataset_name in dataset_names:
    dataset = form_dataset(dataset_name)
    device = f'cuda:{args.cuda}'
    member_embed_list = {}
    non_member_embed_list = {}
    for set_name in ["train", "test"]:
        cleaned_data, orig_indices = clean_dataset(dataset[set_name], dataset_name)
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
            #pdb.set_trace()
            torch.cuda.synchronize()
            try:
                with torch.no_grad():
                    outputs = model(**tokenized_inputs, labels=target_labels, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states
                for layer_index in range(len(hidden_states)):
                    if layer_index not in member_embed_list:
                        member_embed_list[layer_index] = []
                        non_member_embed_list[layer_index] = []
                    context_embedding = hidden_states[layer_index][0].mean(0).squeeze()
                    if set_name == "train" and len(member_embed_list) < args.samples:
                        member_embed_list[layer_index].append(context_embedding.cpu())
                    elif set_name == "test" and len(non_member_embed_list) < args.samples:
                        non_member_embed_list[layer_index].append(context_embedding.cpu())
            except:
                continue
    for layer_index in range(len(hidden_states)):
        member_embed_array = np.array(member_embed_list[layer_index])
        non_member_embed_array = np.array(non_member_embed_list[layer_index])
                    # Concatenate for PCA
        all_embed_array = np.vstack([member_embed_array, non_member_embed_array])
        labels = np.array([1] * len(member_embed_array) + [0] * len(non_member_embed_array))
        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(all_embed_array)
        # Separate the results
        pca_member_embed = pca_result[labels == 1]
        pca_non_member_embed = pca_result[labels == 0]
        # Plotting
        plt.figure(figsize=(14, 8))
        plt.scatter(pca_member_embed[:, 0], pca_member_embed[:, 1], c='blue', label='Member Text', alpha=0.5)
        plt.scatter(pca_non_member_embed[:, 0], pca_non_member_embed[:, 1], c='red', label='Non-Member Text',
                    alpha=0.5)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('PCA of Member and Non-Member Embeddings')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'embedding_figure/PCA_{dataset_name}_{args.model_size}_{layer_index}.png')
        plt.show()
        labels = np.array([1] * len(member_embed_array) + [0] * len(non_member_embed_array))
        X = np.vstack((member_embed_array, non_member_embed_array))
        db_index = davies_bouldin_score(X, labels)
        silhouette_avg = silhouette_score(X, labels)
        calinski_index = calinski_harabasz_score(X, labels)
        results_df = results_df._append({'Dataset Name': dataset_name,
                                        'Layer Index': layer_index,
                                        'DB Index': db_index,
                                        'Silhouette Score': silhouette_avg,
                                        'Calinski Harabasz Index': calinski_index},
                                       ignore_index=True)
results_df.to_csv(f"embedding_results/{args.model_size}_embedding_result.csv", index=False)


