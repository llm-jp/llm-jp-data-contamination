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
    with open(fpath, 'w+', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
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

def obtain_instruction(dataset_name, split_name):
    if dataset_name in ["jnli", "jsicker", "jamp"]:
        guided_chat = [
            {"role": "system",
             "content": f"### 指示：次の文1は、{dataset_name}データセットの{split_name}分割から提供されています。\n文1の後にあるラベルは、文1と文2の間の論理的な関係を示します。\n文2はデータセット内のインスタンスと完全に一致する必要があります。\nデータセットに表示された通りに、文2を完成させてください。以上の情報を使って、文2だけを出力してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "### 指示：文1に基づいて文2を完成させてください。次のラベルは、文1と文2の間の論理的な関係を示します。以上の情報を使って、文2だけを出力してください。"},
            {"role": "user", "content": ""},
        ]
        guided_chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '\\n\\n### 指示:\\n' + message['content'] }}{% elif message['role'] == 'system' %}{{ '### 指示：jnliデータセットのtrain分割から文1が提供される。データセットに現れた文2を完成させなさい。文2はデータセットのサンプルと正確に一致しなければならないです。' }}{% elif message['role'] == 'assistant' %}{{ '\\n\\n### 応答:\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\\n\\n### 応答:\\n' }}{% endif %}{% endfor %}"
        general_chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '\\n\\n### 指示:\\n' + message['content'] }}{% elif message['role'] == 'system' %}{{ '### 指示：以下のラベルが文1と文2の論理的関係を示すように、文1を基に文2を完成させる。' }}{% elif message['role'] == 'assistant' %}{{ '\\n\\n### 応答:\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\\n\\n### 応答:\\n' }}{% endif %}{% endfor %}"
        return guided_chat, general_chat, guided_chat_template, general_chat_template

    
    