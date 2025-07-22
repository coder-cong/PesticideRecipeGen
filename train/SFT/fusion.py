from transformers import pipeline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

# --- 1. 定义模型和Adapter的路径 ---
# 基础模型路径，可以是Hugging Face Hub上的ID，也可以是本地路径
# 例如： "meta-llama/Llama-2-70b-hf" 或 "/path/to/your/local/llama-72b"
base_model_path = "/data/models/qwen2.5-72B"

# 你训练好的LoRA adapter的路径
lora_adapter_path = "/root/projs/LLaMA-Factory/src/saves/Qwen2.5-72B-Instruct/lora/train_2025-07-05-14-48-49"

# 融合后模型的保存路径
merged_model_save_path = "/data/models/trained/qwen2.5-72B-sft"

print(f"基础模型: {base_model_path}")
print(f"LoRA Adapter: {lora_adapter_path}")
print(f"融合后模型将保存至: {merged_model_save_path}")

# --- 2. 加载模型和Tokenizer ---
# 对于72B这样的大模型，内存是一个巨大的挑战。我们有几种策略：

# 策略A：使用bfloat16并自动进行设备映射 (推荐，如果你的GPU显存+CPU内存足够大)
# `device_map="auto"` 会自动将模型层分配到可用的GPU和CPU内存中，甚至可以利用磁盘空间。
# `torch_dtype=torch.bfloat16` 可以将模型大小减半，并且在现代GPU上性能很好。
print("正在加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 策略B：使用4-bit量化加载 (如果内存非常有限)
# 如果上面的方法导致OOM (Out of Memory)，你可以用4-bit量化来加载基础模型。
# PEFT在融合时会自动处理反量化、合并和重新量化的过程。
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )
# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_path,
#     quantization_config=quantization_config,
#     device_map="auto",
#     trust_remote_code=True,
# )

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path, trust_remote_code=True)

# --- 3. 加载LoRA Adapter并应用到基础模型上 ---
# 这会返回一个PeftModel对象，它在基础模型之上“虚拟地”应用了adapter
print("正在加载并应用LoRA Adapter...")
lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

# --- 4. 融合Adapter到模型中 ---
# 这是核心步骤。`merge_and_unload()`会执行合并操作，
# 然后将模型恢复为标准的Transformers模型，并卸载PEFT相关的模块。
print("开始融合Adapter...")
merged_model = lora_model.merge_and_unload()
print("融合完成！")

# --- 5. 保存融合后的模型和Tokenizer ---
# 现在 `merged_model` 是一个标准的`AutoModelForCausalLM` 对象，
# 你可以像保存任何Hugging Face模型一样保存它。
print(f"正在保存融合后的模型到 {merged_model_save_path}...")
os.makedirs(merged_model_save_path, exist_ok=True)
merged_model.save_pretrained(merged_model_save_path, safe_serialization=True)
tokenizer.save_pretrained(merged_model_save_path)
print("模型和Tokenizer已成功保存！")

# --- (可选) 6. 测试融合后的模型 ---
# print("测试融合后的模型...")

# pipe = pipeline("text-generation", model=merged_model,
#                 tokenizer=tokenizer, device_map="auto")
# prompt = "The main advantage of merging a LoRA adapter is"
# result = pipe(prompt, max_new_tokens=50)
# print(result[0]['generated_text'])
