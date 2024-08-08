import pandas as pd


model_size = "410m"
dataset_name = "github"

dataset_names = ["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc","pubmed_central", "wikipedia_(en)",
                 "full_pile","WikiMIA64", "WikiMIA128","WikiMIA256", "WikiMIAall"]

df = pd.read_csv(f"feature_results_online/{model_size}.csv", index_col=0)
temp_df = df[df["Dataset Name"] == dataset_name]





