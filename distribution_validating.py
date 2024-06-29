from datasets import load_dataset
import torch

dataset_name = ["test_ArXiv.pt"]
dataset = torch.load(dataset_name[0])
