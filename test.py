import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from utils import *
tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-13b-instruct-full-dolly-ichikara_004_001_single-oasst-oasst2-v2.0")
model = AutoModelForCausalLM.from_pretrained("llm-jp/llm-jp-13b-instruct-full-dolly-ichikara_004_001_single-oasst-oasst2-v2.0", device_map="auto", torch_dtype=torch.bfloat16)
model.generation_config.output_scores = True
model.generation_config.return_dict_in_generate = True
# chat = [
#     {"role": "system", "content": ""},
#     {"role": "user", "content": "１匹の犬が車の後部に乗っています。"},
# ]
chat = [
    {"role": "system", "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"},
    {"role": "user", "content": "自然言語処理とは何か"},
]
text="""
### 指示：jnliデータセットのtrain分割から文1が提供される。データセットに現れた文2を完成させなさい。文2はデータセットのサンプルと正確に一致しなければならないです。

### 指示:
１匹の犬が車の後部に乗っています。
"""
chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '\\n\\n### 指示:\\n' + message['content'] }}{% elif message['role'] == 'system' %}{{ '\\n\\n### 指示:\\n' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\\n\\n### 応答:\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\\n\\n### 応答:\\n' }}{% endif %}{% endfor %}"
tokenized_input = tokenizer.apply_chat_template(chat, chat_template, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(model.device)
#encoded=tokenizer.encode(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        tokenized_input,
        max_new_tokens=100,
        top_k=0,
        top_p=0.95,
        temperature=0.0,
        repetition_penalty=1.05,
    )
# 使用函数计算perplexity
perplexities = calculate_perplexity(output, tokenized_input)

continuation_text = "自然言語処理は、コンピュータが人間の言語を理解し、生成する技術です。"
continuation = tokenizer.encode(continuation_text, return_tensors="pt").squeeze(0).to(model.device)
if continuation[0] == 31:
    continuation = continuation[1:]
memorization_score = calculate_memorization_score(output, tokenized_input, continuation, tokenizer)

# # 打印生成的目标序列
# for i in range(len(perplexities)):
#     print(f"Token {i + 1}: {tokenizer.decode(output.sequences[0][i + tokenized_input.size(-1)])}")
#
# # 示例输出原始和生成序列
# for i in range(len(output.sequences[0])):
#     print(tokenizer.decode(output.sequences[0][i]))
