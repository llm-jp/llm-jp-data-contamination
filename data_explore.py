from datasets import load_dataset

for dataset_name in ["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc", "pubmed_central", "wikipedia_(en)",
                     "full_pile"]:
    dataset = load_dataset("iamgroot42/mimir", dataset_name,
                           split=f"ngram_13_0.2")
    member_length = sum([len(x) for x in dataset["none"]["member"]]) / len(dataset["none"]["member"])
    nonmember_length = sum([len(x) for x in dataset["none"]["nonmember"]]) / len(dataset["none"]["nonmember"])
    print(dataset_name, member_length, nonmember_length)
for dataset_name in ["Ar"]:
    pass

# arxiv 1383.931 1384.555
# dm_mathematics 899.4157014157014 905.3552123552123
# github 1333.3594 1306.6432
# hackernews 1207.648167539267 1194.7057591623036
# pile_cc 1129.495 1126.279
# pubmed_central 1449.179 1445.37
# wikipedia_(en) 1167.0 1167.0
# full_pile 1198.769 1195.7798

