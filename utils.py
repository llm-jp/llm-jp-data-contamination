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
    if dataset_name in ["jnli", "jsick", "jamp","janli"]:
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
    elif dataset_name in ["alt-e-to-j", "alt-j-to-e"]:
        if dataset_name == "alt-e-to-j":
            info = "英語から日本語"
        else:
            info = "日本語から英語"
        guided_chat = [
            {"role": "system",
             "content": f"次の文は、{dataset_name}データセットの{split_name}分割から提供されています。\n原文の前半と原文全体の{info}への翻訳文は提供されています。\nデータセットに表示された通りに、原文の後半を完成させてください。必ずその原文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": f"原文の前半と原文全体の{info}への翻訳文は提供されています。必ずその原文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jemhopqa"]:
        guided_chat = [
                    {"role": "system",
                     "content": f"次の質問文の前半と答えは、{dataset_name}データセットの{split_name}分割から提供されています。\n質問文の下の半分を出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、質問文の下の半分を出力してください。それ以外には何も含めないことを厳守してください。"},
                    {"role": "user", "content": ""},
                ]
        general_chat = [
            {"role": "system",
             "content": f"質問文の半分と答えは提供されいます。\n質問文の下の半分を出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、質問文の下の半分を出力してください。それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jmmlu"]:
        guided_chat = [
                    {"role": "system",
                     "content": f"次の質問文と答えは、{dataset_name}データセットの{split_name}分割から提供されています。\nその質問文と答えを参考し、ABCDという四つの選択肢を出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、選択肢を出力してください。必ずABCDの選択肢のみを出力し、それ以外には何も含めないことを厳守してください。"},
                    {"role": "user", "content": ""},
                ]
        general_chat = [
            {"role": "system",
             "content": f"質問文と答えは提供されています。\nその質問文と答えを参考し、ABCDという四つの選択肢を出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、選択肢を出力してください。必ずABCDの選択肢のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jcommonsenseqa"]:
        guided_chat = [
            {"role": "system",
             "content": f"次の質問文と答えは、{dataset_name}データセットの{split_name}分割から提供されています。\nその質問文と答えを参考し、この答えと質問文に合う01234という五つの選択肢を順番に出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、選択肢を出力してください。必ず01234の選択肢のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": f"質問文と答えは提供されています。\nその質問文と答えを参考し、この答えと質問文に合う01234という五つの選択肢を順番に出力してください。回答の他には何も含めないことを厳守してください。\nデータセットに表示された通りに、選択肢を出力してください。必ず01234の選択肢のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jsts"]:
        guided_chat = [
            {"role": "system",
             "content": f"次の文1は、{dataset_name}データセットの{split_name}分割から提供されています。\n文1の後にある数字は、文1と文2の間の類似度を示します。0.0に近いほど文ペアの意味が異なり、5.0に近いほど文ペアの意味が似ていることを表しています。\n文2はデータセット内のインスタンスと完全に一致する必要があります。\nデータセットに表示された通りに、文2を完成させてください。文1とその類似度を使って、必ず文2のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "文1の後にある数字は、文1と文2の間の類似度を示します。0.0に近いほど文ペアの意味が異なり、5.0に近いほど文ペアの意味が似ていることを表しています。\n文1とその類似度を使って、必ず文2のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["niilc"]:
        guided_chat = [
            {"role": "system",
             "content": f"次の質問に対する答えと質問文の前半は、{dataset_name}データセットの{split_name}分割から提供されています。\nデータセットに表示された通りに、質問文の後半を完成させてください。\nその文はデータセット内のインスタンスと完全に一致する必要があります。必ず質問文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "質問に対する答えと質問文の前半は提供されています。\n必ず質問文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["mawps"]:
        guided_chat = [
            {"role": "system",
             "content": f"次の計算問題に対する答えと質問文の前半は、{dataset_name}データセットの{split_name}分割から提供されています。\データセットに表示された通りに、質問文の後半を完成させてください。必ずその質問文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "計算問題に対する答えと質問文の前半は提供されています。\n必ずその質問文の後半のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jsquad"]:
        guided_chat = [
            {"role": "system",
             "content": f"文章と文章に対する答えは、{dataset_name}データセットの{split_name}分割から提供されています。\nデータセットに表示された通りに、その文章と答えに合う質問文を書いてください。必ず質問文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "文章と文章に対する答えは提供されています。\nその文章と答えに合う質問文を書いてください。必ず質問文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["jsem"]:
        guided_chat = [
            {"role": "system",
             "content": f"前提と、前提と仮説の関係の答えは、{dataset_name}データセットの{split_name}分割から提供されています。\nその答えはyes、no、unknown、undefの中からの答えと提供されています\nデータセットに表示された通りに、仮説文を完成させてください。必ず仮説文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "前提と、前提と仮説の関係の答えはyes、no、unknown、undefから提供されています\nデータセットに表示された通りに、仮説文を書いてください。必ず仮説文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template
    elif dataset_name in ["chabsa"]:
        guided_chat = [
            {"role": "system",
             "content": f"与えられた文章の前半と、全体の文章から抽出された固有表現で書かれたターゲットの名前と、それぞれの名前に対するpositive、neutral、negativeの極性は、{dataset_name}データセットの{split_name}分割から提供されています。\nデータセットに表示された通りに、未完成の文章の後半を書いてください。必ず文章の後半の文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        general_chat = [
            {"role": "system",
             "content": "与えられた文章の前半と、全体の文章から抽出された固有表現で書かれたターゲットの名前と、それぞれの名前に対するpositive、neutral、negativeの極性は提供されています\n未完成の文章の後半を書いてください。必ず文章の後半の文のみを出力し、それ以外には何も含めないことを厳守してください。"},
            {"role": "user", "content": ""},
        ]
        return guided_chat, general_chat, chat_template

def formalize_input(dataset_name,guided_chat, general_chat, inst_type, example):
    if dataset_name in ["jnli", "jsick", "jamp", "janli"]:
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
    elif dataset_name in ["alt-e-to-j", "alt-j-to-e"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        if dataset_name == "alt-e-to-j":
            sent1 = " ".join(example['input'].split(" ")[:len(example['input'].split(" ")) // 2])
            sent1 = sent1.strip()
            label = example['output']
            sent2 = " ".join(example['input'].split(" ")[len(example['input'].split(" "))//2:])
            sent2= sent2.strip()
        else:
            sent1 = example['input'][:len(example['input']) // 2]
            sent1 = sent1.strip()
            label = example['output']
            sent2 = example['input'][len(example['input']) // 2:]
            sent2 = sent2.strip()
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"原文:{sent1}\n翻訳文:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"原文:{sent1}\n翻訳文:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["jemhopqa", "niilc", "mawps"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1 = example['input'][:len(example['input'])//2]
        label = example['output']
        sent2 = example['input'][len(example['input'])//2:]
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"質問文:{sent1}\n答え:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"質問文:{sent1}\n答え:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["jsts"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1, sent2 = example['input'].split('\n')
        label = example['output']
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"文1:{sent1.strip()}\n類似度:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"文1:{sent1.strip()}\n類似度:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["jcommonsenseqa", "jmmlu"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1 = example['input'].split("\n")[0]
        label = example['output']
        sent2 = example['input'].split("\n")[1]
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"質問文:{sent1}\n答え:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"質問文:{sent1}\n答え:{label}\n"
        return chat, sent1, sent2, instruction
    # elif dataset_name in ["niilc"]:
    #     instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
    #     question = f"答え: {example['output']}"
    #     answer = f"回答: {example['intput']}"
    #     if inst_type == 'guided_instruction':
    #         chat = guided_chat
    #         chat[1]["content"] = f"{question}\n"
    #     else:
    #         chat = general_chat
    #         chat[1]["content"] = f"{question}\n"
    #     return chat, question, answer, instruction
    # elif dataset_name in ["mawps"]:
    #     instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
    #     question = f"答え: {example['output']}"
    #     answer = f"回答: {example['intput']}"
    #     if inst_type == 'guided_instruction':
    #         chat = guided_chat
    #         chat[1]["content"] = f"{question}\n"
    #     else:
    #         chat = general_chat
    #         chat[1]["content"] = f"{question}\n"
    #     return chat, question, answer, instruction
    elif dataset_name in ["jsquad"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1, sent2 = example['input'].split('\n')
        label = example['output']
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"文章:{sent1}\n答え:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"文章:{sent1}\n答え:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["jsem"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1, sent2 = example['input'].split('\n')
        label = example['output']
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"前提:{sent1}\n前提と仮説の関係の答え:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"前提:{sent1}\n前提と仮説の関係の答え:{label}\n"
        return chat, sent1, sent2, instruction
    elif dataset_name in ["chabsa"]:
        instruction = guided_chat[0]["content"] if inst_type == 'guided_instruction' else general_chat[0]["content"]
        sent1, sent2 = example['input'][:len(example['input'])//2], example['input'][len(example['input'])//2:]
        label = example['output']
        if inst_type == 'guided_instruction':
            chat = guided_chat
            chat[1]["content"] = f"文章:{sent1}\nターゲットの名前とそれぞれの極性:{label}\n"
        else:
            chat = general_chat
            chat[1]["content"] = f"文章:{sent1}\nターゲットの名前とそれぞれの極性:{label}\n"
        return chat, sent1, sent2, instruction



    
    