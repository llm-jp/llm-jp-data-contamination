import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
        max_new_tokens=10,
        top_k=0,
        top_p=0,
        temperature=0.0,
        repetition_penalty=1.05,
    )

print(tokenizer.decode(output))
import torch.nn.functional as F
sequences = output.sequences[0][len(tokenized_input[0]):].unsqueeze(0)
scores = torch.cat(output.scores, dim=0)

shift_logits = scores[:-1, :].contiguous()
shift_labels = sequences[:, 1:].contiguous().view(-1)

loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels, reduction='mean')

perplexities = []
for i in range(shift_labels.size(0)):
    current_logits = scores[i, :].unsqueeze(0)
    current_label = shift_labels[i].unsqueeze(0)
    loss = F.cross_entropy(current_logits, current_label, reduction='mean')
    perplexity = loss
    perplexities.append(perplexity.item())
    #perplexity = torch.exp(loss)
# 输出每个step的困惑度
for i, perp in enumerate(perplexities):
    print(f"Step {i}: 困惑度 = {perp}")

