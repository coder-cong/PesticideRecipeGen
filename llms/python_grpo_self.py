import os
import re
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F
from tqdm import tqdm
import logging
import random
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from util.get_dataset import  read_gsmk8k
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 系统提示词
SYSTEM_PROMPT = """
请先仔细思考这个数学问题，然后给出正确答案。
按照如下格式生成：
<思考>
在这里写出详细的解题步骤和推理过程
</思考>
<答案>
在这里只写出最终的数值答案
</答案>
"""


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def extract_answer(text):
    """从生成的文本中提取答案"""
    # 尝试从<答案>标签中提取
    answer_pattern = r"<答案>(.*?)</答案>"
    match = re.search(answer_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 尝试从<answer>标签中提取
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 如果没有找到标签，尝试获取最后一行或句子
    lines = text.strip().split('\n')
    if lines:
        return lines[-1].strip()

    return text.strip()


def format_prompt(tokenizer, prompt, max_length=256):
    """格式化提示为模型输入"""
    # 将提示转换为字符串
    if isinstance(prompt, list):
        formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
    else:
        formatted_prompt = prompt

    # 编码并截断
    encoded = tokenizer(
        formatted_prompt,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return encoded, formatted_prompt


def generate_responses(model, tokenizer, prompt_tokens, num_generations=4, max_new_tokens=100, temperature=1.0):
    """生成多个候选响应"""
    model.eval()  # 设置为评估模式

    # 将输入移动到模型所在设备
    input_ids = prompt_tokens.input_ids.to(model.device)
    attention_mask = prompt_tokens.attention_mask.to(model.device)
    prompt_length = input_ids.shape[1]

    responses = []

    # 生成多个响应
    with torch.no_grad():
        for _ in range(num_generations):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )

            # 提取生成的文本（不包括提示部分）
            generated_ids = outputs[0, prompt_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(generated_text)

    return responses, prompt_length


def compute_rewards(responses, expected_answer):
    """计算每个响应的奖励"""
    rewards = []
    log_info = []

    for response in responses:
        # 提取答案
        extracted_answer = extract_answer(response)

        # 计算奖励 - 简单的精确匹配
        reward = 1.0 if extracted_answer.strip() == expected_answer.strip() else 0.0

        rewards.append(reward)
        log_info.append({
            "extracted_answer": extracted_answer,
            "reward": reward
        })

    return rewards, log_info


def compute_log_probs(model, tokenizer, prompt_tokens, response, prompt_length, device):
    """计算模型对响应的对数概率"""
    # 编码响应
    response_tokens = tokenizer(response, return_tensors="pt").input_ids.to(device)

    # 构建完整输入序列 (提示+响应)
    full_input_ids = torch.cat([prompt_tokens.input_ids, response_tokens], dim=1).to(device)
    full_attention_mask = torch.ones_like(full_input_ids).to(device)

    # 计算模型输出
    outputs = model(input_ids=full_input_ids, attention_mask=full_attention_mask)
    logits = outputs.logits[:, prompt_length - 1:-1, :]

    # 计算对数概率
    log_probs = F.log_softmax(logits, dim=-1)

    # 获取目标token的对数概率
    target_ids = full_input_ids[:, prompt_length:]
    token_log_probs = torch.gather(
        log_probs, 2, target_ids.unsqueeze(-1)
    ).squeeze(-1)

    # 计算序列平均对数概率
    seq_log_prob = token_log_probs.sum(dim=1) / target_ids.shape[1]

    return seq_log_prob


def compute_grpo_loss(policy_logprobs, reference_logprobs, rewards, beta=0.1):
    """
    计算GRPO损失 - 基于DeepSeek论文的方法

    参数:
        policy_logprobs: 形状为[n]的张量，策略模型对n个生成答案的对数概率
        reference_logprobs: 形状为[n]的张量，参考模型对相同n个答案的对数概率
        rewards: 形状为[n]的张量，每个答案的奖励值
        beta: KL散度的权重系数

    返回:
        GRPO损失
    """
    # 转换为张量(如果不是)
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, device=policy_logprobs.device)

    # 1. 计算奖励的标准化优势值
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # 添加小值避免除零

    # 2. 计算策略和参考模型之间的KL散度 (对数概率之差)
    '''
    关键这个地方实际上参考网络的梯度是不参与计算的，应该减掉
    '''
    kl_divergence = policy_logprobs - reference_logprobs

    # 3. 计算GRPO损失：-E[(log P_policy - log P_ref) * advantage]
    '''
        这个地方实际上的意义是 log(policy)*adavantages
    '''
    policy_loss = -(kl_divergence * advantages).mean()

    # 4. 添加KL正则化项以限制整体偏离
    kl_reg = beta * kl_divergence.mean()

    # 5. 最终GRPO损失
    loss = policy_loss + kl_reg

    return loss


def main(args):
    """主训练函数"""
    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化tensorboard
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载策略模型
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
    )

    # 应用LoRA (如果启用)
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        policy_model = get_peft_model(policy_model, lora_config)
        policy_model.print_trainable_parameters()

    # 加载参考模型 (与策略模型相同的初始权重，但不训练)
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
    )

    # 将模型移动到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model = policy_model.to(device)
    reference_model = reference_model.to(device)

    # 冻结参考模型
    for param in reference_model.parameters():
        param.requires_grad = False

    # 加载数据集
    dataset = read_gsmk8k(args.data_dir,sys_prompt=SYSTEM_PROMPT, split=args.split)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)  # 批次大小为1，因为我们生成多个答案

    # 设置优化器
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 设置学习率调度器
    num_training_steps = len(data_loader) * args.num_epochs
    lr_scheduler = get_scheduler(
        name=args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps
    )

    # 开始训练
    global_step = 0

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_rewards = []

        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch in progress_bar:
            # 获取问题和答案
            prompt = batch['prompt'][0]  # 取第一个元素，因为batch_size=1
            question = batch['question'][0]
            expected_answer = batch['answer'][0]

            # 格式化提示
            prompt_tokens, formatted_prompt = format_prompt(tokenizer, prompt)

            # 生成多个候选答案
            responses, prompt_length = generate_responses(
                policy_model,
                tokenizer,
                prompt_tokens,
                num_generations=args.num_generations,
                max_new_tokens=args.max_completion_length,
                temperature=args.temperature
            )

            # 计算奖励
            rewards, log_info = compute_rewards(responses, expected_answer)

            # 如果所有奖励都为0，可以选择跳过这个样本
            if sum(rewards) == 0 and random.random() < 0.5:  # 50%概率跳过全0奖励样本
                continue

            # 计算策略模型的对数概率
            policy_logprobs = []
            for response in responses:
                policy_logprob = compute_log_probs(
                    policy_model,
                    tokenizer,
                    prompt_tokens,
                    response,
                    prompt_length,
                    device
                )
                policy_logprobs.append(policy_logprob)

            # 计算参考模型的对数概率
            reference_logprobs = []
            with torch.no_grad():
                for response in responses:
                    reference_logprob = compute_log_probs(
                        reference_model,
                        tokenizer,
                        prompt_tokens,
                        response,
                        prompt_length,
                        device
                    )
                    reference_logprobs.append(reference_logprob)

            # 转换为张量
            policy_logprobs = torch.cat(policy_logprobs)
            reference_logprobs = torch.cat(reference_logprobs)
            rewards_tensor = torch.tensor(rewards, device=device)

            # 计算GRPO损失
            loss = compute_grpo_loss(
                policy_logprobs,
                reference_logprobs,
                rewards_tensor,
                beta=args.beta
            )

            # 反向传播和优化
            loss.backward()

            # 梯度裁剪
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # 更新统计信息
            epoch_loss += loss.item()
            epoch_rewards.extend(rewards)

            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_reward': sum(rewards) / len(rewards) if rewards else 0
            })

            # 记录到tensorboard
            tb_writer.add_scalar('loss', loss.item(), global_step)
            tb_writer.add_scalar('avg_reward', sum(rewards) / len(rewards) if rewards else 0, global_step)
            tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

            # 日志记录
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # 找出最佳响应
                best_idx = np.argmax(rewards)

                logger.info(
                    f"Step {global_step} | Loss: {loss.item():.4f} | Avg Reward: {sum(rewards) / len(rewards):.4f}")
                logger.info(f"问题: {question}")
                logger.info(f"预期答案: {expected_answer}")
                logger.info(f"最佳响应: {responses[best_idx]}")
                logger.info(f"最佳提取答案: {log_info[best_idx]['extracted_answer']}")

                # 记录标准化优势值
                advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
                logger.info(f"标准化优势值: {advantages.cpu().numpy()}")

            # 保存检查点
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)

                if args.use_lora:
                    policy_model.save_pretrained(checkpoint_dir)
                else:
                    policy_model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

                logger.info(f"保存检查点到 {checkpoint_dir}")

            global_step += 1

        # Epoch结束统计
        avg_epoch_loss = epoch_loss / len(data_loader)
        avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0

        logger.info(f"Epoch {epoch + 1} 完成: Avg Loss = {avg_epoch_loss:.4f}, Avg Reward = {avg_epoch_reward:.4f}")

        # 保存每个epoch的模型
        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch + 1}")
        os.makedirs(epoch_dir, exist_ok=True)

        if args.use_lora:
            policy_model.save_pretrained(epoch_dir)
        else:
            policy_model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)

        logger.info(f"保存epoch模型到 {epoch_dir}")

    # 训练结束，保存最终模型
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)

    if args.use_lora:
        policy_model.save_pretrained(final_dir)
    else:
        policy_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

    logger.info(f"保存最终模型到 {final_dir}")
    logger.info("训练完成")

    # 关闭tensorboard写入器
    tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO训练 - DeepSeek方法")

    # 数据和模型参数
    parser.add_argument("--data_dir", type=str, default="/media/iiap/25df545d-3a24-4466-b58d-f96c46b9a3bf/数据集/gsm8k_chinese/data", help="GSM8K Chinese数据集目录")
    parser.add_argument("--split", type=str, default="train", help="使用的数据集分割 ('train' 或 'test')")
    parser.add_argument("--model_path", type=str, default="/media/iiap/25df545d-3a24-4466-b58d-f96c46b9a3bf/LargeModel/Qwen2.5-0.5B-Instruct", help="预训练模型路径")
    parser.add_argument("--output_dir", type=str, default="./grpo_trained", help="输出目录")

    # 训练超参数
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=1, help="训练轮次")
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成的样本数量")
    parser.add_argument("--max_completion_length", type=int, default=100, help="最大回复长度")
    parser.add_argument("--temperature", type=float, default=1.0, help="生成温度")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--scheduler", type=str, default="cosine", help="学习率调度器类型")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪最大范数")
    parser.add_argument("--beta", type=float, default=0.1, help="GRPO损失中的KL散度权重")

    # LoRA参数
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA丢弃率")

    # 日志和保存参数
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500, help="保存检查点的步数")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--bf16", action="store_true", help="是否使用bfloat16精度")

    args = parser.parse_args()
    main(args)