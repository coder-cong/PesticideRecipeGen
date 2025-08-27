import os
import argparse
from typing import List
import torch as torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM,  BitsAndBytesConfig
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataset import GRPODataset


# def load_dataset_test():
#     data_path = "/root/projs/PesticideRecipeGen/data/distill/distill_data_alpaca.json"
#     model_path = "/data/models/qwen2.5-7B"
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     dataset = GRPODataset(data_path=data_path, tokenizer=tokenizer)
#     for data in dataset:
#         print(data)


def init_ddp():
    dist.init_process_group("nccl")
    return (int(os.environ['RANK']), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]))


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


def generate_batch_fsdp(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> List[str]:
    """
    使用手动前向传播循环为一批提示并行生成响应，兼容 FSDP 模型。
    Args:
        model: FSDP 包裹的 Transformer 模型。
        tokenizer: 对应的分词器。
        prompts: 一个字符串列表，每个字符串是一个独立的提示。
        max_new_tokens: 每个响应生成的最大新 token 数量。
        temperature: 控制生成随机性的温度。值越小，生成越确定。
        top_p: Top-p (nucleus) 采样的累积概率阈值。
    Returns:
        一个字符串列表，包含每个提示对应的生成响应。
    """
    model.eval()
    # FSDP 模型已经分布在各个 GPU 上，我们只需要获取当前 rank 的 device
    device = model.device
    # 1. 设置 Tokenizer 并进行批处理编码
    # **关键**: 对于自回归生成，必须使用左填充！
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size, prompt_length = input_ids.shape
    # 2. 初始化生成状态
    generated_tokens = input_ids
    past_key_values = None

    # 跟踪哪些序列还没有生成结束符
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=device)
    # 【修改】初始化一个张量来存储每个响应的长度
    response_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 3. 准备模型输入
            # 如果有 KV 缓存，我们只需要输入最后一个 token
            if past_key_values is not None:
                input_ids_step = generated_tokens[:, -1:]
            else:
                input_ids_step = generated_tokens
            outputs = model(
                input_ids=input_ids_step,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )

            # 4. 从 logits 中采样下一个 token
            next_token_logits = outputs.logits[:, -1, :]

            # 应用 temperature
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # 计算概率并应用 top-p 采样
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = _sample_top_p(probs, p=top_p)
            # 5. 处理已经完成的序列
            # 如果一个序列已经完成，我们用 pad_token 填充，而不是新采样的 token
            next_token = next_token * unfinished_sequences.unsqueeze(
                1) + tokenizer.pad_token_id * (1 - unfinished_sequences.unsqueeze(1))

            # 6. 更新状态以进行下一次迭代
            generated_tokens = torch.cat(
                [generated_tokens, next_token], dim=-1)

            # 更新 attention mask (只为未完成的序列添加 1)
            attention_mask = torch.cat(
                [attention_mask, unfinished_sequences.unsqueeze(1)],
                dim=1
            )

            past_key_values = outputs.past_key_values
            # 【修改】在更新 unfinished_sequences 之前，累加长度
            # 只有仍在生成的序列 (unfinished_sequences值为1) 的长度会增加
            response_lengths += unfinished_sequences

            # 7. 检查哪些序列刚刚生成了结束符
            unfinished_sequences = unfinished_sequences & (
                next_token.squeeze(1) != tokenizer.eos_token_id)

            # 如果所有序列都已完成，提前退出
            if not torch.any(unfinished_sequences):
                break
    # 8. 解码并返回结果
    # 移除输入提示部分
    output_ids = generated_tokens[:, prompt_length:]
    generated_texts = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True)

    return generated_texts, (response_lengths-1).cpu().tolist()


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained("/data/models/qwen2.5-0.5B")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("/data/models/qwen2.5-0.5B")
    dataset = GRPODataset(
        "/root/projs/PesticideRecipeGen/data/distill/distill_data_alpaca.json", tokenizer)
    prompts = [dataset[0]]*5
    responses, lens = generate_batch_fsdp(
        model, tokenizer, prompts, 1024, 1.0, 0.7)
    print(
        [f"len:{lens[i]},response:{responses[i]}" for i in range(len(responses))])
