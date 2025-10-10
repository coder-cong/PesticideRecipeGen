import re
import torch
import gc
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType
import os
import openai
from typing import List, Dict

# 农药配方生成的系统提示
SYSTEM_PROMPT = """
你是一个专业的农药配方专家。请根据用户的需求，生成安全、有效的农药配方。
请按照以下格式回答：

>Thinking...
>分析病虫害类型、作物特点、环境条件等因素
>

<answer>
配方名称：
主要成分：
配比：
使用方法：
注意事项：
</answer>
"""

# 创建农药配方数据集


def create_pesticide_dataset():
    """创建农药配方训练数据集"""
    sample_data = [
        {
            "question": "请为水稻稻飞虱防治提供一个有效的农药配方",
            "answer": """配方名称：水稻稻飞虱防治配方
主要成分：吡虫啉、噻嗪酮
配比：25%吡虫啉可湿性粉剂20-30g + 25%噻嗪酮可湿性粉剂40-50g，兑水15公斤
使用方法：在稻飞虱发生初期喷雾，7-10天后视虫情再喷一次
注意事项：避免在强风和高温时段施药，注意轮换用药防止抗性"""
        },
        {
            "question": "小麦蚜虫防治需要什么农药配方？",
            "answer": """配方名称：小麦蚜虫综合防治配方
主要成分：啶虫脒、氟啶虫胺腈
配比：3%啶虫脒乳油30-40ml + 10%氟啶虫胺腈悬浮剂10-15ml，兑水15公斤
使用方法：在蚜虫发生初期至盛期喷雾防治，间隔7-10天
注意事项：避免在开花期使用，保护天敌，交替用药"""
        },
        {
            "question": "玉米螟虫害防治配方推荐",
            "answer": """配方名称：玉米螟生物防治配方
主要成分：苏云金杆菌、氯虫苯甲酰胺
配比：16000IU/mg苏云金杆菌可湿性粉剂100-150g + 5%氯虫苯甲酰胺悬浮剂20-30ml，兑水15公斤
使用方法：在玉米螟卵孵化盛期至幼虫3龄前喷雾
注意事项：选择阴天或傍晚施药，避免紫外线照射影响药效"""
        },
        {
            "question": "番茄晚疫病防治农药配方",
            "answer": """配方名称：番茄晚疫病预防治疗配方
主要成分：烯酰吗啉、代森锰锌
配比：50%烯酰吗啉可湿性粉剂15-20g + 80%代森锰锌可湿性粉剂100-150g，兑水15公斤
使用方法：预防性用药7-10天一次，发病初期5-7天一次
注意事项：雨前雨后及时用药，注意药剂轮换使用"""
        },
        {
            "question": "苹果树红蜘蛛防治配方",
            "answer": """配方名称：苹果红蜘蛛综合防治配方
主要成分：阿维菌素、哒螨灵
配比：1.8%阿维菌素乳油2000-3000倍液 + 15%哒螨灵乳油2000-2500倍液
使用方法：在红蜘蛛发生初期喷雾，重点喷洒叶背面
注意事项：避免高温时段施药，保护捕食螨等天敌"""
        }
    ]

    return Dataset.from_list(sample_data)


def process_data(data):
    """处理农药配方数据"""
    def format_sample(example):
        return {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': example['question']}
            ],
            'answer': example['answer']
        }

    return data.map(format_sample)

# 奖励函数1：格式检查奖励


def format_reward(completions, **kwargs):
    """检查回答格式是否正确"""
    try:
        responses = [completion[0]["content"] for completion in completions]
        rewards = []

        for response in responses:
            if not response or not isinstance(response, str):
                rewards.append(0.0)
                continue

            reward = 0.0

            # 检查必要的格式标记
            if ">Thinking..." in response:
                reward += 0.2
            if "<answer>" in response and "</answer>" in response:
                reward += 0.3

            # 检查配方必要字段
            required_fields = ["配方名称", "主要成分", "配比", "使用方法", "注意事项"]
            for field in required_fields:
                if field in response:
                    reward += 0.1

            rewards.append(min(reward, 1.0))

        return rewards
    except Exception as e:
        print(f"格式奖励计算错误: {e}")
        return [0.0] * len(completions)

# 奖励函数2：内容质量奖励


def content_quality_reward(completions, **kwargs):
    """基于规则的内容质量评估"""
    try:
        responses = [completion[0]["content"] for completion in completions]
        rewards = []

        for response in responses:
            if not response or not isinstance(response, str):
                rewards.append(0.0)
                continue

            reward = 0.0

            # 检查是否包含具体的农药成分名称
            pesticide_keywords = [
                "吡虫啉", "噻嗪酮", "啶虫脒", "氟啶虫胺腈", "苏云金杆菌",
                "氯虫苯甲酰胺", "烯酰吗啉", "代森锰锌", "阿维菌素", "哒螨灵"
            ]
            for keyword in pesticide_keywords:
                if keyword in response:
                    reward += 0.1
                    break

            # 检查是否包含具体的配比信息
            if re.search(r'\d+[gml%]', response):
                reward += 0.2

            # 检查是否包含使用方法
            method_keywords = ["喷雾", "施药", "兑水", "倍液"]
            for keyword in method_keywords:
                if keyword in response:
                    reward += 0.1
                    break

            # 检查是否包含安全注意事项
            safety_keywords = ["注意", "避免", "防止", "保护"]
            for keyword in safety_keywords:
                if keyword in response:
                    reward += 0.2
                    break

            rewards.append(min(reward, 1.0))

        return rewards
    except Exception as e:
        print(f"内容质量奖励计算错误: {e}")
        return [0.0] * len(completions)

# 奖励函数3：安全性检查奖励


def safety_reward(completions, **kwargs):
    """检查农药配方的安全性"""
    try:
        responses = [completion[0]["content"] for completion in completions]
        rewards = []

        for response in responses:
            if not response or not isinstance(response, str):
                rewards.append(0.0)
                continue

            reward = 0.5  # 基础分

            # 检查是否提到了安全用药
            safety_mentions = [
                "注意事项", "安全", "防护", "避免", "禁止",
                "间隔期", "轮换", "抗性", "天敌"
            ]
            safety_count = sum(
                1 for mention in safety_mentions if mention in response)
            reward += min(safety_count * 0.1, 0.5)

            # 惩罚可能不安全的建议
            unsafe_keywords = ["随意", "大量", "频繁", "混用"]
            for keyword in unsafe_keywords:
                if keyword in response:
                    reward -= 0.2

            rewards.append(max(0.0, min(reward, 1.0)))

        return rewards
    except Exception as e:
        print(f"安全性奖励计算错误: {e}")
        return [0.0] * len(completions)


def get_optimized_config_for_3090():
    """针对8卡3090优化的配置"""
    config = {
        "per_device_train_batch_size": 1,  # 3090显存限制
        "gradient_accumulation_steps": 4,   # 增加梯度累积
        "num_generations": 4,               # 减少生成数量
        "generation_batch_size": 2,         # 减少生成批次大小
        "dataloader_num_workers": 4,        # 多进程加载数据
    }
    return config


def clear_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # 使用Qwen2.5-7B模型
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # 或者你的本地路径

    print("加载模型...")

    # 配置LoRA以减少显存占用
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj",
                        "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 应用LoRA
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("创建农药配方数据集...")
    data = create_pesticide_dataset()
    processed_data = process_data(data)
    print(f"训练数据大小: {len(processed_data)}")

    output_dir = "pesticide_grpo_output"

    # 获取针对3090的优化配置
    opt_config = get_optimized_config_for_3090()

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,  # 7B模型使用较小学习率
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        max_prompt_length=512,      # 增加提示长度
        max_completion_length=512,  # 增加完成长度
        num_train_epochs=2,
        save_steps=50,
        eval_steps=50,
        max_grad_norm=1.0,
        log_on_each_node=False,
        use_vllm=False,
        report_to="tensorboard",
        remove_unused_columns=False,

        # 多卡训练配置
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,

        **opt_config  # 应用优化配置
    )

    print("初始化训练器...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward,
            content_quality_reward,
            safety_reward
        ],
        args=training_args,
        train_dataset=processed_data,
    )

    # 清理内存后开始训练
    clear_memory()
    print("开始训练...")
    trainer.train()

    print("保存模型...")
    trainer.save_model(output_dir)
    print("训练完成!")
