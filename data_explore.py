from datasets import load_dataset

dataset = load_dataset("iamgroot42/mimir", "arxiv",
                               split="ngram_13_0.2")
member_length = sum([len(x) for x in dataset["member"]])/len(dataset["member"])
nonmember_length = sum([len(x) for x in dataset["nonmember"]])/len(dataset["nonmember"])
print(member_length, nonmember_length)