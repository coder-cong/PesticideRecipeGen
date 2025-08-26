from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
import os
import json


base_model_path = "/data/models/qwen2.5-72B"
lora_adapter_path = "/root/projs/LLaMA-Factory/src/saves/Qwen2.5-72B-Instruct/lora/train_2025-07-05-14-48-49"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # 使用NF4类型进行量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 在计算时，权重会反量化为bfloat16，以保持精度
    bnb_4bit_use_double_quant=True,  # 使用双重量化
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

print("正在加载基础模型到多张GPU上...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,  # 使用bfloat16以节省显存并提高性能 (如果你的GPU支持)
    quantization_config=bnb_config,  # 如果要使用4-bit量化，取消这一行的注释

    # -------- 关键参数 --------
    device_map="auto",
    # --------------------------

    # 如果遇到"out of memory"错误，可以尝试这个参数来减少峰值内存使用
    # low_cpu_mem_usage=True
)

# --- 5. 加载并融合LoRA Adapter ---
# PEFT库会自动处理已经分布在多卡上的基础模型
print("\n正在加载LoRA Adapter...")
model = PeftModel.from_pretrained(
    base_model,          # 传入已经加载到多卡上的模型
    lora_adapter_path,     # LoRA adapter的路径或ID
    is_trainable=False   # 我们是用于推理，所以设置为False
)
print("LoRA Adapter加载并融合完成。")

# --- 6. 进行推理 ---
# 注意：即使模型分布在多卡，输入数据通常也需要放到第一个设备上
# `device_map="auto"`通常会将模型的输入层(word_embeddings)放在`cuda:0`上
prompt = "What is the capital of France?"
inputs = tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True)
print(inputs)


streamer = TextIteratorStreamer(
    tokenizer, skip_prompt=True, skip_special_tokens=True)

generation_kwargs = dict(
    inputs=inputs,
    streamer=streamer,
    max_new_tokens=1024,
    temperature=0.5
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for new_text in streamer:
    print(
        f"data: {json.dumps({'id': 'chatcmpl-123', 'object': 'chat.completion.chunk', 'model': 'your-model-name', 'created': 1726114622, 'choices': [{'index': 0, 'delta': {'content': new_text}, 'finish_reason': None}]})}\n\n")
