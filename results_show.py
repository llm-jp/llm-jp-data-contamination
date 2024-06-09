import argparse
import json


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
                        default="llm-jp",
                        help="the name of model")
    parser.add_argument("--num_samples",
                        type=int,
                        default=15,
                        help="the number of samples")
    args = parser.parse_args()
    if args.dataset_name == "all":
        dataset_names = ["alt-e-to-j", "alt-j-to-e","chabsa", "jamp", "janli",
                         "jcommonsenseqa", "jemhopqa", "jmmlu", "jnli", "jsem",
                         "jsick", "jsquad","jsts", "mawps", "niilc"]
        for dataset_name in dataset_names:
            print(f"Show the results of {dataset_name} in {args.split_name} split")
            with open(f"contamination_result/{dataset_name}/data_contamination_result.jsonl", "r") as f:
                lines = f.readlines()
                for i in lines:
                   print(i)
    else:
        print(f"Show the results of {args.dataset_name} in {args.split_name} split")
        with open(f"contamination_result/{args.dataset_name}/data_contamination_result.jsonl", "r") as f:
            lines = f.readlines()
            for i in lines:
                print(i)
