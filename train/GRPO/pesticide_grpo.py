# 测试grpo训练循环，使用7B模型测试
import time
from dataset import GRPODataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import copy
import inspect
import os
import re
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sequence, Sized
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union
from peft import LoraConfig

from util import SYSTEM_PROMPT, USER_CONTENT, request_dmx_api, request_hanka_api

import datasets
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available

from trl import GRPOTrainer
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    disable_dropout_in_model,
    entropy_from_logits,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
from typing import Dict, List, Tuple, Optional


def _sample_top_p(probs, p):
    """
    Top-P (Nucleus) Sampling 的辅助函数。
    支持批处理。
    Args:
        probs: A `[batch_size, vocab_size]` tensor of probabilities.
        p: The cumulative probability threshold.
    Returns:
        A `[batch_size, 1]` tensor of the sampled token indices.
    """
    # 1. 对概率进行排序并计算累积和
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    # 2. 创建一个 mask 来移除累积概率超过 p 的 token
    # 我们也保留第一个 token，以防 p 太小
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0

    # 3. 重新归一化概率
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    # 4. 从修改后的分布中采样
    next_token = torch.multinomial(probs_sort, num_samples=1)

    # 5. 获取原始的 token 索引
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def _generate_parallel_responses(
    local_rank: int,
    model,
    tokenizer,
    prompt: str,  # 单个字符串
    num_current_batch_generations: int,  # 当前批次需要生成的数量 (例如 num_parallel)
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> Tuple[List[str], List[int]]:
    """
    为单个提示生成 num_current_batch_generations 次响应，每次作为一个批次推理。
    启用 KV-Cache 以提高推理效率。
    """
    model.eval()
    device = f"cuda:{local_rank}"
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 将单个提示复制 num_current_batch_generations 次
    prompts_list = [prompt] * num_current_batch_generations
    # 编码批处理的提示
    inputs = tokenizer(
        prompts_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size, prompt_length = input_ids.shape
    generated_tokens = input_ids  # 最初包含完整的prompt_ids
    # 初始化 unfinished_sequences 跟踪每个序列是否完成
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=device)
    # 存储每个响应的生成长度
    response_length = torch.zeros(
        batch_size, dtype=torch.long, device=device)
    past_key_values = None  # 初始化 KV Cache

    with torch.no_grad():
        for i in range(max_new_tokens):
            if i == 0:  # 第一次迭代，传入完整的 input_ids
                model_input_ids = input_ids
                model_attention_mask = attention_mask
            else:  # 后续迭代，只传入上一步生成的 token
                model_input_ids = next_token  # `next_token` 是 [batch_size, 1]
                # attention_mask 每次都会扩展一个 token，所以需要传递完整的
                # 这里 `attention_mask` 已经是包含了 `generated_tokens` 的全部长度
                # 模型的 forward 会根据 `model_input_ids` 和 `attention_mask` 计算新的 KV cache
                # 注意：如果 `model_input_ids` 长度为1，attention_mask 也只需要处理到新加入的token
                # 但更简单且兼容性更好的方法是，每次 `attention_mask` 也只传入最新的部分
                # 然而，如果模型内部的 KV cache 处理逻辑依赖于传入的 attention_mask 长度与 input_ids 长度匹配，
                # 那么这里传入完整的 `attention_mask` 就不对了。
                # 正确做法是，attention_mask也每次更新，长度与generated_tokens匹配。
                # 但更标准的hf transformers模型在使用KV cache时，attention_mask会是累积的。
                # 下面使用累积的 attention_mask。
            outputs = model(
                input_ids=model_input_ids,
                attention_mask=model_attention_mask,  # 传入累积的 attention_mask
                past_key_values=past_key_values,  # 传入上一步的 KV Cache
                use_cache=True,  # 明确启用 KV Cache
            )

            # 总是取最后一个 token 的 logits
            next_token_logits = outputs.logits[:, -1, :]
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = _sample_top_p(probs, p=top_p)

            # 对于已经完成的序列，将生成的 token 替换为 pad_token_id
            next_token = next_token * unfinished_sequences.unsqueeze(
                1) + tokenizer.pad_token_id * (1 - unfinished_sequences.unsqueeze(1))
            # 将新生成的 token 添加到已生成的序列中
            generated_tokens = torch.cat(
                [generated_tokens, next_token], dim=-1)
            # 更新 attention_mask：对于尚未完成的序列，其 attention_mask 增加一个 1
            # 注意：这里我们更新的 attention_mask 是用于下一次迭代的，因为它代表了 `generated_tokens` 的 mask
            model_attention_mask = torch.cat(  # 这就是累积的 attention_mask
                [model_attention_mask, unfinished_sequences.unsqueeze(1)],
                dim=1
            )

            # 更新 KV Cache
            past_key_values = outputs.past_key_values  # 从模型输出中获取更新后的 KV Cache
            # 更新响应长度，只计算尚未完成的序列
            response_length += unfinished_sequences
            # 检查哪些序列已完成 (生成 EOS token)
            unfinished_sequences = unfinished_sequences & (
                next_token.squeeze(1) != tokenizer.eos_token_id)
            # 如果所有序列都已完成，则提前退出循环
            if not torch.any(unfinished_sequences):
                break

    # 提取生成的补全部分（排除原始提示）
    output_ids = generated_tokens[:, prompt_length:]
    all_generated_texts: List[str] = []
    all_response_lengths: List[int] = []
    # 遍历批次中的每个生成结果进行解码
    for i in range(batch_size):
        generated_text = tokenizer.decode(
            output_ids[i], skip_special_tokens=True)
        all_generated_texts.append(generated_text)
        all_response_lengths.append(response_length[i].item())
    return all_generated_texts, all_response_lengths


def generate_batch_fsdp(
    local_rank: int,
    model,
    tokenizer,
    prompt: str,  # 单个字符串
    num_generations: int,  # 需要生成的总补全数量
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    num_parallel: int = 1,  # 控制并行生成提示的数量 (一个批次中包含多少个重复的提示)
) -> Tuple[List[str], List[int]]:
    """
    为单个提示生成 num_generations 次响应，每次使用 num_parallel 个并行批次。

    Args:
        local_rank: 当前进程的局部排名（用于CUDA设备分配）。
        model: 模型实例。
        tokenizer: 分词器实例。
        prompt: 单个字符串提示。
        num_generations: 需要为该提示生成的总补全数量。
        max_new_tokens: 每个补全的最大新 token 数量。
        temperature: 采样温度。
        top_p: Top-P 采样阈值。
        num_parallel: 在一个批次中并行处理的重复提示数量。
                      例如，如果 num_generations=8, num_parallel=2，
                      则会进行 4 次模型调用，每次传入 2 个相同的提示。
    Returns:
        一个包含所有生成的文本列表和对应长度列表的元组。
    """
    all_generated_texts: List[str] = []
    all_response_lengths: List[int] = []

    # 计算需要进行多少个并行批次处理
    num_batches_needed = (num_generations + num_parallel - 1) // num_parallel

    for i in range(num_batches_needed):
        # 计算当前批次需要生成的实际数量
        # 最后一个批次可能少于 num_parallel
        num_current_batch_generations = min(
            num_parallel, num_generations - len(all_generated_texts))

        if num_current_batch_generations <= 0:
            break  # 已经生成了足够的数量，退出
        if local_rank == 0:
            print(
                f"Generating batch {i+1}/{num_batches_needed} "
                f"({num_current_batch_generations} parallel responses) "
                f"for prompt: '{prompt[:50]}...'"
            )
        # 调用新的辅助函数生成一个批次的响应
        generated_texts_batch, response_lengths_batch = _generate_parallel_responses(
            local_rank=local_rank,
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            num_current_batch_generations=num_current_batch_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        all_generated_texts.extend(generated_texts_batch)
        all_response_lengths.extend(response_lengths_batch)

    return all_generated_texts, all_response_lengths


def compute_rewards(
    prompt: str,
    responses: List[str],
    max_retries: int = 10,  # 新增参数: 最大重试次数
    retry_delay_seconds: int = 2  # 新增参数: 重试间隔时间
) -> Optional[List[float]]:  # 返回类型改为 Optional[List[float]]
    """
    计算给定提示和每个响应的奖励得分。
    通过调用大模型API，将提示和响应拼接成完整输入，由大模型进行打分。
    如果API调用失败，会进行重试。如果超过最大重试次数仍失败，则返回None。
    Args:
        prompt: 原始的提示字符串。
        responses: 大模型为该提示生成的补全响应列表。
        max_retries: 每个API请求的最大重试次数。
        retry_delay_seconds: 每次重试前的等待时间（秒）。
    Returns:
        一个浮点数列表，表示每个响应的奖励得分。
        如果任何一个响应的打分在重试后仍失败，则整个函数返回 None。
    """
    rewards: List[float] = []
    import requests
    import json
    import time
    # 循环处理每个响应
    for i, response_text in enumerate(responses):
        current_reward = None  # 用于存储当前响应的奖励
        # 0到max_retries，所以总共max_retries+1次尝试
        for attempt in range(max_retries + 1):
            try:
                system_msg = SYSTEM_PROMPT
                user_msg = USER_CONTENT.format(
                    prompt=prompt, response_text=response_text)
                response_data = request_hanka_api(system_msg, user_msg)
                # --- 解析API响应以提取奖励 ---
                model_output = response_data
                if model_output is not None and model_output != "":
                    match = re.search(r'总分:\s*(\d+)', model_output)
                    if match:
                        score = float(match.group(1))
                        score = max(0.0, min(10.0, score))
                        current_reward = score
                        print(
                            f"  Response {i+1} Score (Attempt {attempt+1}/{max_retries+1}): {score} - {model_output}")
                        break  # 成功获取奖励，跳出重试循环
                    else:
                        print(
                            f"  Warning: Could not extract score from API response for Response {i+1} (Attempt {attempt+1}/{max_retries+1}): {model_output}")
                else:
                    print(
                        f"  Warning: No valid choices in API response for Response {i+1} (Attempt {attempt+1}/{max_retries+1}): {response_data}")
            except requests.exceptions.Timeout:
                print(
                    f"  API Request Timeout for Response {i+1} (Attempt {attempt+1}/{max_retries+1}). Retrying...")
            except requests.exceptions.RequestException as e:
                print(
                    f"  Error calling API for Response {i+1} (Attempt {attempt+1}/{max_retries+1}): {e}. Retrying...")
            except json.JSONDecodeError as e:
                print(
                    f"  Error decoding JSON from API for Response {i+1} (Attempt {attempt+1}/{max_retries+1}): {e}. Retrying...")
            except Exception as e:
                print(
                    f"  An unexpected error occurred for Response {i+1} (Attempt {attempt+1}/{max_retries+1}): {e}. Retrying...")

            # 如果不是最后一次尝试，则等待后重试
            if attempt < max_retries:
                time.sleep(retry_delay_seconds)
            else:
                print(
                    f"  Max retries ({max_retries}) exceeded for Response {i+1}. Skipping this response.")
                current_reward = None  # 达到最大重试次数仍失败，设置为None

        # 将当前响应的奖励添加到列表中
        # 如果任何一个响应打分失败 (current_reward 为 None)，则整个函数应该返回 None
        if current_reward is None:
            print(
                f"  Failed to get reward for Response {i+1} after all retries. Returning None for the entire batch.")
            return None  # 任何一个失败，则整个批次失败
        else:
            rewards.append(current_reward)
    return rewards


def compute_log_probs(model, is_ref, tokenizer, prompt: str, response: str, device):
    """计算模型对响应的对数概率"""
    import torch.nn.functional as F
    # 编码响应
    response_tokens = tokenizer(
        response, return_tensors="pt").input_ids.to(device)
    prompt_tokens = tokenizer(
        prompt, return_tensors="pt").input_ids.to(device)
    prompt_length = prompt_tokens.shape[1]

    # 构建完整输入序列 (提示+响应)
    full_input_ids = torch.cat(
        [prompt_tokens, response_tokens], dim=1).to(device)
    full_attention_mask = torch.ones_like(full_input_ids).to(device)

    # 计算模型输出
    if is_ref:
        outputs = reference_model_infer(
            model, full_input_ids, full_attention_mask)
    else:
        outputs = model(input_ids=full_input_ids,
                        attention_mask=full_attention_mask)

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


def reference_model_infer(policy_model, full_input_ids, full_attention_mask):
    with policy_model.disable_adapter():
        outputs = policy_model(input_ids=full_input_ids,
                               attention_mask=full_attention_mask)
    return outputs


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
    advantages = (rewards - rewards.mean()) / \
        (rewards.std() + 1e-8)  # 添加小值避免除零

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


model_path = "/data/lyl/models/qwen2.5-0.5B"
data_path = "/data/lyl/projs/PesticideRecipeGen/data/distill/distill_data_alpaca.json"
# 1. 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_path)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj",
                    "o_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)
# 2. 加载数据集和optimizer
dataset = GRPODataset(data_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=1)
# Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95),
                              eps=1e-5, weight_decay=0.1, fused=True)
gradient_accumulation_steps = 8
# 3. 复制训练循环，跳过生成补全步骤直接测试后续步骤
rank = 0
device = f"cuda:{rank}"
num_generations = 8
max_completion_length = 1024
temperature = 1.0
for epoch in range(5):
    model.train()
    # 这里因为有可能有的batch会跳过，另外使用一个变量来计数有效的batch数
    effective_batch_idx = 0
    for batch_idx, batch in enumerate(dataloader):

        accumulate_grads = (
            effective_batch_idx+1) % gradient_accumulation_steps == 0

        # 获取问题和答案
        prompt = batch[0]  # 取第一个元素，因为batch_size=1

        # 1. 生成多个候选答案
        responses, prompt_lens = generate_batch_fsdp(
            rank,
            model,
            tokenizer,
            prompt,
            num_generations=num_generations,
            max_new_tokens=max_completion_length,
            temperature=temperature,
            num_parallel=num_generations//2
        )
        if rank == 0:
            print(time.strftime(
                '%Y-%m-%d %H:%M:%S', time.localtime()))
            print("完成推理")

        # 2. 计算奖励得分
        rewards = compute_rewards(prompt, responses)
        if rewards is None or sum(rewards) == 0:
            continue
        if rank == 0:
            print(rewards)
        # 3. 计算policy和ref模型logprobs
        # 计算策略模型的对数概率
        if rank == 0:
            print(time.strftime(
                '%Y-%m-%d %H:%M:%S', time.localtime()))
            print("开始计算logprob")
        policy_logprobs = []
        for response in responses:
            policy_logprob = compute_log_probs(
                model,
                False,
                tokenizer,
                prompt,
                response,
                device
            )
            policy_logprobs.append(policy_logprob)

        # 计算参考模型的对数概率
        reference_logprobs = []
        with torch.no_grad():
            for response in responses:
                reference_logprob = compute_log_probs(
                    model,
                    True,
                    tokenizer,
                    prompt,
                    response,
                    device
                )
                reference_logprobs.append(reference_logprob)
        if rank == 0:
            print(time.strftime(
                '%Y-%m-%d %H:%M:%S', time.localtime()))
            print("完成计算logprob")
        # 转换为张量
        policy_logprobs = torch.cat(policy_logprobs)
        reference_logprobs = torch.cat(reference_logprobs)
        rewards_tensor = torch.tensor(rewards, device=device)
        # 4. 计算loss
        loss = compute_grpo_loss(
            policy_logprobs,
            reference_logprobs,
            rewards_tensor,
        )
        effective_batch_idx += 1

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps

        loss.backward()

        # Record loss
        bs = batch['input_ids'].shape[0]
        ddp_loss[0] += loss.item() * bs * gradient_accumulation_steps
        ddp_loss[1] += bs

        # Step the optimizer (w/ gradient accumulation)
        if accumulate_grads:
            optimizer.step()
            optimizer.zero_grad()
            # avoid overhead when lr is constant.
            if lr_scheduler is not None:
                lr_scheduler.step()
            progress_bar.update(1)

        # Delete the output so more memory frees up before the next forward pass
        output = None
        loss = None

        # Stop logging memory (first iter)
        if args['profile_memory'] and batch_idx == 0 and rank == 0 and epoch == 0:
            torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
            torch.cuda.memory._record_memory_history(
                enabled=None)  # Stop recording

        # Log loss every gradient update steps
        if accumulate_grads:
            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
            if rank == 0:
                log_loss = ddp_loss[0] / ddp_loss[1]
                if lr_scheduler is not None:
                    log_lr = lr_scheduler.get_last_lr()[0]
                else:
                    log_lr = args["lr"]
                update_progress_bar(
                    progress_bar, epoch, log_loss, log_lr, rank)
                if args["log_to"] == 'wandb':
                    logger.log({"loss": log_loss, "lr": log_lr}, rank)
            ddp_loss = torch.zeros(2).to(local_rank)

        if rank == 0 and args['verbose']:
            print(f"Batch idx {batch_idx}")

        prof.step()

        # Primarily for debugging
        if args["max_steps"] > 0 and batch_idx > args["max_steps"]:
            if rank == 0:
                print("Max steps reached, skipping rest of epoch")
            break

    # Print + log peak memory usage for the whole fourth step of training
    if epoch == 0 and (rank == 0 or args['verbose']):
        peak_allocated_memory = torch.cuda.max_memory_allocated(
            local_rank)
        peak_reserved_memory = torch.cuda.max_memory_reserved(
            local_rank)
        memory_stats.append(
            f"Rank {rank}: Peak allocated memory: {peak_allocated_memory/2**30:.2f} GiB")
        memory_stats.append(
            f"Rank {rank}: Peak reserved memory:  {peak_reserved_memory/2**30:.2f} GiB")
        if args["log_to"] == 'wandb':
            logger.log(
                {"memory/allocated_peak": peak_allocated_memory}, rank)
            logger.log(
                {"memory/reserved_peak": peak_reserved_memory}, rank)
