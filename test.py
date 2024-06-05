import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-13b-v2.0")
model = AutoModelForCausalLM.from_pretrained("llm-jp/llm-jp-13b-v2.0", device_map="auto", torch_dtype=torch.bfloat16)
text = "自然言語処理とは何か"
tokenized_input = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(model.device)
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