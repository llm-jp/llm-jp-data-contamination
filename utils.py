import os
import json

NUM_SAMPLES = 15
TASK_NAME = ["nli-task"]
SPLIT_NAME = ["train", "dev"]
DATASET_NAME = {
    "nli": "WNLI"
}
SAVE_PATHS = {
    "wnli": {
        "train": {
            "raw": "data/nli-task/wnli/train.jsonl",
            "gpt_response": "data/nli-task/wnli/gpt_response_train.jsonl"
        },
        "dev": {
            "raw": "data/nli-task/wnli/dev.jsonl",
            "gpt_response": "data/nli-task/wnli/gpt_response_dev.jsonl"
        }
    }
}

def save_jsonl(data, fpath):
    with open(fpath, 'w+') as f:
        json.dump(data, f, indent=2)
        
def load_json(fpath):
    with open(fpath) as f:
        return json.load(f)

def clean_text(text):
    words_list = []
    text = text.lower().replace("sentence 2:", '').replace(".", '')
    for sentence in text.split(','):
        words = sentence.replace('["', '')\
                .replace('\\n', '')\
                .replace('"]', '')\
                .replace("['", '')\
                .replace("]", '').split(' ')
        words_list += words
    if '' in words_list: 
        for _ in range(words_list.count('')):
            words_list.remove('')
    
    return words_list


    
    