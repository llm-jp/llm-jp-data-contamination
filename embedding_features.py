import os
import pdb
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from utils import form_dataset, batched_data_with_indices, clean_dataset, load_model_and_tokenizer, get_dataset_list, obtain_dataset
import argparse
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import roc_auc_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--model_size", type=str, default="1b")
parser.add_argument("--dataset_name", type=str, default="Pile-CC", choices=["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                     "pubmed_central", "wikipedia_(en)", "full_pile", "temporalarxiv","all"])
parser.add_argument("--cuda", type=int, default=1, help="cuda device")
parser.add_argument("--skip_calculation", type=str, default="True")
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--gradient_collection", type=str, default=False)
args = parser.parse_args()


dataset_names = get_dataset_list(args.dataset_name)
model, tokenizer = load_model_and_tokenizer(args)
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = model.generation_config.eos_token_id
os.makedirs(f"embedding_results_online/", exist_ok=True)
csv_file_path = f"embedding_results_online/{args.model_size}_embedding_result.csv"
results_df = pd.DataFrame(
    columns=['Dataset Name', 'Layer Index', 'DB Index',
             'Silhouette Score', 'Calinski Harabasz Index'])
for dataset_name in dataset_names:
    dataset = obtain_dataset(dataset_name)
    device = f'cuda:{args.cuda}'
    member_embed_list = {}
    non_member_embed_list = {}
    for set_name in ["member", "nonmember"]:
        cleaned_data, orig_indices = clean_dataset(dataset[set_name])
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
            with torch.no_grad():
                outputs = model(**tokenized_inputs, labels=target_labels, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            for layer_index in range(len(hidden_states)):
                if layer_index not in member_embed_list:
                    member_embed_list[layer_index] = []
                    non_member_embed_list[layer_index] = []
                context_embedding = hidden_states[layer_index][0].mean(0).squeeze()
                if set_name == "member" and len(member_embed_list) < args.samples:
                    member_embed_list[layer_index].append(context_embedding.cpu())
                elif set_name == "nonmember" and len(non_member_embed_list) < args.samples:
                    non_member_embed_list[layer_index].append(context_embedding.cpu())
    for layer_index in tqdm(range(len(hidden_states))):
        if os.path.exists(csv_file_path):
            results_df = pd.read_csv(csv_file_path)
            if ((results_df["Dataset Name"] == dataset_name) & (results_df["Layer Index"] == layer_index)).any():
                print(f"Skipping training for {dataset_name} at layer {layer_index} as previous results are found.")
                continue
        member_embed_array = torch.stack(member_embed_list[layer_index])
        non_member_embed_array = torch.stack(non_member_embed_list[layer_index])
                    # Concatenate for PCA
        all_embed_array = torch.cat([member_embed_array, non_member_embed_array])
        labels = np.array([1] * len(member_embed_array) + [0] * len(non_member_embed_array))
        # Perform PCA
        pca = PCA(n_components=2)
        all_embed_array = all_embed_array.float().cpu().numpy()
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
        #X = np.vstack((member_embed_array, non_member_embed_array))
        db_index = davies_bouldin_score(all_embed_array, labels)
        silhouette_avg = silhouette_score(all_embed_array, labels)
        calinski_index = calinski_harabasz_score(all_embed_array, labels)
        results_df = results_df._append({'Dataset Name': dataset_name,
                                        'Layer Index': layer_index,
                                        'DB Index': db_index,
                                        'Silhouette Score': silhouette_avg,
                                        'Calinski Harabasz Index': calinski_index},
                                       ignore_index=True)
        results_df.to_csv(f"embedding_results_online/{args.model_size}_embedding_result.csv", index=False)


