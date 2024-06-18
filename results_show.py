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
                temp_line = ""
                for line in lines:
                    # lineの前後の空白文字を削除
                    line = line.strip()
                    if line.startswith("{") and temp_line:
                        try:
                            # JSONに変換して結果リストに追加
                            data = json.loads(temp_line)
                            data["bleurt_guided_prompt"] = data["average bleurt score"][0]
                            data["bleurt_naive_prompt"] = data["average bleurt score"][1]
                            data["rougeL_guided_prompt"] = data["rougeL score"][0]
                            data["rougeL_naive_prompt"] = data["rougeL score"][1]
                            results.append(data)
                        except json.JSONDecodeError as e:
                            print(f"跳过无效的JSON行: {temp_line}")
                            print(f"错误: {e}")
                        temp_line = line
                    else:
                        temp_line += " " + line

                    # 最後のJSONオブジェクトを追加
                if temp_line:
                    try:
                        data = json.loads(temp_line)
                        results.append(data)
                    except json.JSONDecodeError as e:
                        print(f"跳过无效的JSON行: {temp_line}")
                        print(f"错误: {e}")
    else:
        print(f"Show the results of {args.dataset_name} in {args.split_name} split")
        with open(f"contamination_result/{args.dataset_name}/{args.split_name}/data_contamination_result.jsonl", "r") as f:
            lines = f.readlines()
            temp_line = ""
            for line in lines:
                # lineの前後の空白文字を削除
                line = line.strip()
                if line.startswith("{") and temp_line:
                    try:
                        # JSONに変換して結果リストに追加
                        data = json.loads(temp_line)
                        data["bleurt_guided_prompt"] = data["average bleurt score"][0]
                        data["bleurt_naive_prompt"] = data["average bleurt score"][1]
                        data["rougeL_guided_prompt"] = data["rougeL score"][0]
                        data["rougeL_naive_prompt"] = data["rougeL score"][1]
                        results.append(data)
                    except json.JSONDecodeError as e:
                        print(f"跳过无效的JSON行: {temp_line}")
                        print(f"错误: {e}")
                    temp_line = line
                else:
                    temp_line += " " + line

                # 最後のJSONオブジェクトを追加
            if temp_line:
                try:
                    data = json.loads(temp_line)
                    results.append(data)
                except json.JSONDecodeError as e:
                    print(f"跳过无效的JSON行: {temp_line}")
                    print(f"错误: {e}")
df = pd.json_normalize(results)

# 删除原始的‘average bleurt score’和‘rougeL score’列
df.drop(columns=["average bleurt score", "rougeL score"], inplace=True)
markdown_table = df.to_markdown()

# Display the DataFrame
print(markdown_table)