import argparse
import json
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",
                        type=str,
                        default="jnli",
                        choices=["alt-e-to-j", "alt-j-to-e","chabsa", "jamp", "janli",
                                 "jcommonsenseqa", "jemhopqa", "jmmlu", "jnli", "jsem",
                                 "jsick", "jsquad","jsts", "mawps", "niilc", "all"],
                        help="the name of dataset")
    parser.add_argument("--split_name",
                        type=str,
                        default="train",
                        choices=["train", "dev", "test"],
                        help="the partition of dataset")
    parser.add_argument("--model",
                        type=str,
                        default="llm-jp-v1",
                        help="the name of model")
    parser.add_argument("--num_samples",
                        type=int,
                        default=15,
                        help="the number of samples")
    args = parser.parse_args()
    results = []
    if args.dataset_name == "all":
        dataset_names = ["alt-e-to-j", "alt-j-to-e","chabsa", "jamp", "janli",
                         "jcommonsenseqa", "jemhopqa", "jmmlu", "jnli", "jsem",
                         "jsick", "jsquad","jsts", "mawps", "niilc"]
        for dataset_name in dataset_names:
            print(f"Show the results of {dataset_name} in {args.split_name} split")
            with open(f"contamination_result/{dataset_name}/{args.split_name}/data_contamination_result.jsonl", "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()  # 前後の空白文字を削除
                    if line:  # 如果行不是空的
                        try:
                            # JSONに変換
                            data = json.loads(line)
                            # 将数据添加到结果列表
                            results.append(data)
                        except json.JSONDecodeError as e:
                            print(f"pass error: {line}")
                            print(f"error: {e}")
    else:
        print(f"Show the results of {args.dataset_name} in {args.split_name} split")
        with open(f"contamination_result/{args.dataset_name}/{args.split_name}/data_contamination_result.jsonl", "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()  # 前後の空白文字を削除
                if line:  # 如果行不是空的
                    try:
                        # JSONに変換
                        data = json.loads(line)
                        # 将数据添加到结果列表
                        results.append(data)
                    except json.JSONDecodeError as e:
                        print(f"pass error: {line}")
                        print(f"error: {e}")
df = pd.DataFrame(results)
markdown_table = df.to_markdown()

# Display the DataFrame
print(markdown_table)