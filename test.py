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

# 打印生成的目标序列
for i in range(len(perplexities)):
    print(f"Token {i + 1}: {tokenizer.decode(output.sequences[0][i + tokenized_input.size(-1)])}")

# 示例输出原始和生成序列
for i in range(len(output.sequences[0])):
    print(tokenizer.decode(output.sequences[0][i]))
# #print(tokenizer.decode(output))
# # sequences = output.sequences[0][len(tokenized_input[0]):].unsqueeze(0)
# # scores = torch.cat(output.scores, dim=0)
# # shift_logits = scores[:-1, :].contiguous()
# # shift_labels = sequences[:, 1:].contiguous().view(-1)
# # perplexities = []
# # for i in range(shift_labels.size(0)):
# #     current_logits = scores[i, :].unsqueeze(0)
# #     current_label = shift_labels[i].unsqueeze(0)
# #     loss = F.cross_entropy(current_logits, current_label, reduction='mean')
# #     perplexity = loss
# #     perplexities.append(perplexity.item())
# #     #perplexity = torch.exp(loss)
# # # 输出每个step的困惑度
# # for i, perp in enumerate(perplexities):
# #     print(f"Step {i}: 困惑度 = {perp}")
#
# logits = torch.stack(output.scores, dim=1)[0]  # (sequence_length, vocab_size)
# generated_seq = output.sequences[0]  # (sequence_length_with_prompt + generated_tokens)
#
# # 提取生成部分的目标序列
# target_seq = generated_seq[tokenized_input.size(-1):]
#
# # 确保logits和目标序列对齐
# assert logits.size(0) == target_seq.size(0), "Logits and target sequence length must match."
# # 计算交叉熵损失和困惑度
# perplexities = []
# with torch.no_grad():
#     for i in range(logits.size(0)):
#         current_logits = logits[i, :].unsqueeze(0)  # (1, vocab_size)
#         current_label = target_seq[i].unsqueeze(0)  # (1,)
#         loss = F.cross_entropy(current_logits, current_label, reduction='none')
#         perplexity = torch.exp(loss).item()
#         perplexities.append(perplexity)
# # 输出每个step的困惑度
# for i, perp in enumerate(perplexities):
#     print(f"Step {i + 1}: 困惑度 = {perp}")
#     print(tokenizer.decode(target_seq[i]))
# print(tokenizer.decode(output[0][0]))
# for i in range(len(target_seq)):
#     print(tokenizer.decode(target_seq[i]))