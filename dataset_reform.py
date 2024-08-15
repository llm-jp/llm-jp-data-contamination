import datasets

lenth = 150
length_gap = 100
data_name = "ArXiv"
data = datasets.load_from_disk(f"/home/bchen/llm-jp-data-contamination/filtered_dataset/{lenth}_{lenth+length_gap}/{data_name}")
lenth = 250
data2 = datasets.load_from_disk(f"/home/bchen/llm-jp-data-contamination/filtered_dataset/{lenth}_{lenth+length_gap}/{data_name}")
