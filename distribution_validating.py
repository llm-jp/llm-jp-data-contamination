from datasets import load_dataset
import torch

dataset_name = ["ArXiv", "DM Mathematics", "Enron Emails", "EuroParl", "FreeLaw", "Github", "Gutenberg (PG-19)",
                "HackerNews", "NIH ExPorter", "PhilPapers", "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                "Ubuntu IRC", "USPTO Backgrounds", "Wikipedia (en)"]
split_name = ["train", "validation", "test"]
