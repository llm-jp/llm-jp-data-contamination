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
    chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '\\n\\n### 指示:\\n' + message['content'] }}{% elif message['role'] == 'system' %}{{ '\\n\\n### 指示:\\n' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\\n\\n### 応答:\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\\n\\n### 応答:\\n' }}{% endif %}{% endfor %}"
    if dataset_name in ["jnli", "jsicker", "jamp"]:
        guided_chat = [
            {"role": "system",
             "content": f"次の文1は、{dataset_name}データセットの{split_name}分割から提供されています。\n文1の後にあるラベルは、文1と文2の間の論理的な関係を示します。\n文2はデータセット内のインスタンスと完全に一致する必要があります。\nデータセットに表示された通りに、文2を完成させてください。必ず文2のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "以下のラベルが文1と文2の論理的関係を示すように、文1を基に文2を完成させる。\n必ず文2のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["alt-e-to-j.json", "alt-j-to-e.json"]:
        if dataset_name == "alt-e-to-j.json":
            info = "英語から日本語"
        else:
            info = "日本語から英語"
        guided_chat = [
            {"role": "system",
             "content": f"次の文は、{dataset_name}データセットの{split_name}分割から提供されています。\nその文を{info}へ翻訳してください。\nデータセットに表示された通りに、翻訳を完成させてください。必ず翻訳のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": f"次の文を{info}へ翻訳してください。必ず翻訳のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jemhopqa"]:
        guided_chat = [
                    {"role": "system",
                     "content": f"次の文は、{dataset_name}データセットの{split_name}分割から提供されています。\n質問を入力とし、回答を出力してください。回答の他には何も含めないことを厳守してください。回答が'はい'と'いいえ'で答えることができる場合、'YES'と'NO'で答えてください。\nデータセットに表示された通りに、回答を出力してください。必ず回答のみを出力し、それ以外には何も含めないことを厳守してください。"},
                    {"role": "user", "content": ""},
                ]
        general_chat = [
            {"role": "system",
             "content": f"質問を入力とし、回答を出力してください。回答の他には何も含めないことを厳守してください。回答が'はい'と'いいえ'で答えることができる場合、'YES'と'NO'で答えてください。\nデータセットに表示された通りに、回答を出力してください。必ず回答のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jemhopqa"]:
        guided_chat = [
                    {"role": "system",
                     "content": f"次の文は、{dataset_name}データセットの{split_name}分割から提供されています。\n質問を入力とし、回答を出力してください。回答の他には何も含めないことを厳守してください。回答が'はい'と'いいえ'で答えることができる場合、'YES'と'NO'で答えてください。\nデータセットに表示された通りに、回答を出力してください。必ず回答のみを出力し、それ以外には何も含めないことを厳守してください。"},
                    {"role": "user", "content": ""},
                ]
                general_chat = [
                    {"role": "system",
                     "content": f"質問を入力とし、回答を出力してください。回答の他には何も含めないことを厳守してください。回答が'はい'と'いいえ'で答えることができる場合、'YES'と'NO'で答えてください。\nデータセットに表示された通りに、回答を出力してください。必ず回答のみを出力し、それ以外には何も含めないことを厳守してください。"},
                    {"role": "user", "content": ""},
                ]
        return guided_chat, general_chat, chat_template


def formalize_input(dataset_name,guided_chat, general_chat, inst_type, example):
    if dataset_name in ["jnli", "jsicker", "jamp", "janli"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        procesesd_sent1 = example['input'].split('\n')[0].replace('前提：', '')
        sent1 = f"文1: {procesesd_sent1}"
        sent2 = "文2: "+example['input'].split("\n")[1].replace("仮説：", "")
        label = "含意" if example['output'] == "entailment" else "矛盾" if example['output'] == "contradiction" else "中立"
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\nラベル:{label}\n"

        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\nラベル:{label}\n"

        return chat, sent1, sent2, instruction
    elif dataset_name in ["alt-e-to-j.json", "alt-j-to-e.json"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1 = f"文: {example['input']}"
        sent2 = f"翻訳: {example['output']}"
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"{sent1}\n"

        else:
            chat = general_chat
            chat[1]["content"] = f"{sent1}\n"
        return chat, sent1, sent2, instruction

    
    