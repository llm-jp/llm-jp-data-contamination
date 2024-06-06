import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-13b-instruct-full-dolly-ichikara_004_001_single-oasst-oasst2-v2.0")
model = AutoModelForCausalLM.from_pretrained("llm-jp/llm-jp-13b-instruct-full-dolly-ichikara_004_001_single-oasst-oasst2-v2.0", device_map="auto", torch_dtype=torch.bfloat16)
chat = [
    {"role": "system", "content": "### 指示：jnliデータセットのtrain分割から文1が提供される。データセットに現れた文2を完成させなさい。文2はデータセットのサンプルと正確に一致しなければならないです。"},
    {"role": "user", "content": "１匹の犬が車の後部に乗っています。"},
]
tokenized_input = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        tokenized_input,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.05,
    )[0]
print(tokenizer.decode(output))

