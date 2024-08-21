from datasets import load_dataset
from utils import *
import argparse


def process_data_nli_task(dataset, split_name, dataset_name, save_path):
    instructions = []
    for sentence1, sentence2, label in zip(dataset['sentence1'], dataset['sentence2'], dataset['label']):
        instruction = {
            "guided_instruction": {
            "instruction": f"You are provided with Sentence 1 from the {split_name} split of the {dataset_name} dataset. Finish Sentence 2 as appeared in the dataset. Sentence 2 must exactly match the instance in the dataset.",
            "sentence1": sentence1, 
            "sentence2": sentence2,
            "label": label
            },
            "general_instruction": {
            "instruction": f"Finish Sentence 2 based on Sentence 1, such that the following label shows the logical relationship between Sentence 1 and Sentence 2.",
            "sentence1": sentence1, 
            "sentence2": sentence2,
            "label": label
            }
        }
        instructions.append(instruction)
    save_jsonl(instructions, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_load_name", 
                        type=str,
                        default="wnli",
                        help="the name of dataset")
    parser.add_argument("--dataset_official_name", 
                        type=str,
                        default="WNLI",
                        help="the official name of dataset")
    parser.add_argument("--save_path", 
                        type=str,
                        default=[SAVE_PATHS["wnli"]['train']['raw'],
                                 SAVE_PATHS["wnli"]['dev']['raw']],
                        help="the path to save data")
    args = parser.parse_args()
    
    dataset = load_dataset("glue", args.dataset_load_name)
    train, dev = dataset["train"],  dataset["validation"]
    process_data_nli_task(train, 
                          split_name="train", 
                          dataset_name=args.dataset_official_name, 
                          save_path=args.save_path[0])
    process_data_nli_task(dev, 
                          split_name="dev", 
                          dataset_name=args.dataset_official_name, 
                          save_path=args.save_path[1])