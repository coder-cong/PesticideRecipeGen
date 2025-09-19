import re
import torch
import gc
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType

SYSTEM_PROMPT = """
按照如下格式生成：
>Thinking...
>
>...
>

<answer>
...
</answer>
"""

def process_data(data):
    """处理数据，使用确认存在的字段"""
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question_zh-cn']}
        ],
        'answer': x['answer_only']
    }) 
    return data

def extract_answer(text):
    """优化的答案提取函数"""
    if not text or not isinstance(text, str):
        return ""
    
    # 使用正则表达式提取，支持多行内容
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return ""

def mark_num(text):
    """标记计数函数 - 检查标签出现次数"""
    if not text or not isinstance(text, str):
        return 0.0
        
    reward = 0.0
    # 检查每个标签是否恰好出现一次
    markers = [">Thinking...>", "", "<answer>", "</answer>"]
    
    for marker in markers:
        if text.count(marker) == 1:
            reward += 0.125
            
    return reward

# 生成答案是否正确的奖励
def correctness_reward(prompts, completions, answer, **kwargs):
    """正确性奖励函数"""
    try:
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [extract_answer(r) for r in responses]
        
        # 调试输出（仅第一个样本）
        if len(responses) > 0:
            print(f"问题:\n{prompts[0][-1]['content']}")
            print(f"标准答案:\n{answer[0]}")
            print(f"模型输出:\n{responses[0][:300]}...")
            print(f"提取后的答案:\n{extracted_responses[0]}")
            print("-" * 50)
        
        return [2.0 if str(response).strip() == str(ans).strip() else 0.0 
                for response, ans in zip(extracted_responses, answer)]
    except Exception as e:
        print(f"正确性奖励计算错误: {e}")
        return [0.0] * len(completions)

# 生成答案是否是数字的奖励
def digit_reward(completions, **kwargs):
    """数字奖励函数"""
    try:
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [extract_answer(r) for r in responses]
        return [0.5 if response.strip().replace('.', '').replace('-', '').isdigit() 
                else 0.0 for response in extracted_responses]
    except Exception as e:
        print(f"数字奖励计算错误: {e}")
        return [0.0] * len(completions)

# 严格格式奖励 - 修复正则表达式
def hard_format_reward(completions, **kwargs):
    """严格格式奖励 - 要求精确的换行格式"""
    try:
        # 修复后的正则表达式：支持多行内容
        pattern = r">Thinking...>\n(.*?)\n\n<answer>\n(.*?)\n</answer>"
        responses = [completion[0]["content"] for completion in completions]
        
        rewards = []
        for response in responses:
            # 使用 re.DOTALL 标志支持跨行匹配
            match = re.search(pattern, response, re.DOTALL)
            rewards.append(0.5 if match else 0.0)
            
        return rewards
    except Exception as e:
        print(f"严格格式奖励计算错误: {e}")
        return [0.0] * len(completions)

# 宽松格式奖励 - 修复正则表达式
def soft_format_reward(completions, **kwargs):
    """宽松格式奖励 - 允许灵活的空格和换行"""
    try:
        # 修复后的正则表达式：更宽松的匹配
        pattern = r">Thinking...>\s*(.*?)\s*\s*<answer>\s*(.*?)\s*</answer>"
        responses = [completion[0]["content"] for completion in completions]
        
        rewards = []
        for response in responses:
            # 使用 re.DOTALL 标志支持跨行匹配
            match = re.search(pattern, response, re.DOTALL)
            rewards.append(0.5 if match else 0.0)
            
        return rewards
    except Exception as e:
        print(f"宽松格式奖励计算错误: {e}")
        return [0.0] * len(completions)

# 标记奖励
def mark_reward(completions, **kwargs):
    """标记奖励函数"""
    try:
        responses = [completion[0]["content"] for completion in completions]
        return [mark_num(response) for response in responses]
    except Exception as e:
        print(f"标记奖励计算错误: {e}")
        return [0.0] * len(completions)

def get_optimized_config(output_dir):
    """根据GPU显存自动优化配置"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"检测到GPU显存: {gpu_memory:.1f}GB")
        
        if gpu_memory >= 24:
            config = {
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "num_generations": 8,
                "generation_batch_size": 8,
            }
        elif gpu_memory >= 16:
            config = {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "num_generations": 4,
                "generation_batch_size": 4,
            }
        else:  # 8GB+
            config = {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "num_generations": 2,
                "generation_batch_size": 2,
            }
    else:
        config = {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_generations": 1,
            "generation_batch_size": 1,
        }
    
    print(f"推荐配置: {config}")
    return config

def clear_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    model_name = "/data/lyl/projs/PesticideRecipeGen/train/GRPO/trlGrpo/model/Qwen2.5-0.5B-Instruct"

    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("加载数据集...")
    ds = load_dataset("/data/lyl/projs/PesticideRecipeGen/train/GRPO/trlGrpo/data/gsm8k_chinese", cache_dir="/data")
    data = process_data(ds['train'])
    print(f"训练数据大小: {len(data)}")
    
    output_dir = "output"
    
    # 获取优化配置
    opt_config = get_optimized_config(output_dir)
    
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        max_prompt_length=256,
        max_completion_length=200,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
        report_to="tensorboard",
        dataloader_num_workers=0,
        **opt_config  # 应用优化配置
    )
    
    print("初始化训练器...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            mark_reward,
            soft_format_reward,
            hard_format_reward,
            digit_reward,
            correctness_reward
        ],
        args=training_args,
        train_dataset=data,
    )
    
    # 清理内存后开始训练
    clear_memory()
    print("开始训练...")
    trainer.train()
    
    print("保存模型...")
    trainer.save_model(output_dir)
    print("训练完成!")
