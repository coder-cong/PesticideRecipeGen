# quick_test.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载原始模型
print("加载原始模型...")
orig_tokenizer = AutoTokenizer.from_pretrained("/data/lyl/projs/PesticideRecipeGen/train/GRPO/trlGrpo/model/Qwen2.5-0.5B-Instruct")
orig_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 加载checkpoint模型
print("加载checkpoint模型...")
ckpt_tokenizer = AutoTokenizer.from_pretrained("/data/lyl/projs/PesticideRecipeGen/train/GRPO/trlGrpo/output/checkpoint-1000")
ckpt_model = AutoModelForCausalLM.from_pretrained(
    "output/checkpoint-1000",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 测试问题 - 修复语法错误
question = """
Joy 可以在 20 分钟内读完一本书的 8 页。她读完 120 页需要多少小时？
"""

def generate(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\n问题: {question}")
print("\n原始模型回答:")
print(generate(orig_model, orig_tokenizer, question))

print("\n训练后模型回答:")
print(generate(ckpt_model, ckpt_tokenizer, question))
