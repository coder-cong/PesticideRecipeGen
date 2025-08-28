import os
import argparse
from typing import List, Tuple
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

# 辅助函数：生成单个响应


def _generate_single_response(
    local_rank: int,
    model,
    tokenizer,
    prompt: str,  # 单个字符串
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> Tuple[str, int]:  # 返回单个字符串和长度
    """
    为单个提示生成一个响应，不使用 KV Cache。
    """
    model.eval()
    device = f"cuda:{local_rank}"
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 编码单个提示，batch_size 将为 1
    inputs = tokenizer(
        [prompt],  # 注意这里用列表包裹，因为 tokenizer 期望一个列表
        return_tensors="pt",
        padding=True,  # 实际上对于 batch_size=1 没什么影响
        truncation=True,
    ).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size, prompt_length = input_ids.shape  # batch_size 此时为 1
    generated_tokens = input_ids
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=device)
    response_length = torch.zeros(
        batch_size, dtype=torch.long, device=device)  # 这里是单个响应的长度
    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids_step = generated_tokens

            outputs = model(
                input_ids=input_ids_step,
                attention_mask=attention_mask,
                use_cache=False
            )

            next_token_logits = outputs.logits[:, -1, :]
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = _sample_top_p(probs, p=top_p)

            # 【此处移除打印，因为在循环里每次都打印会很吵，如果需要可以在主函数里打印】
            if local_rank == 0:
                print(next_token)
            next_token = next_token * unfinished_sequences.unsqueeze(
                1) + tokenizer.pad_token_id * (1 - unfinished_sequences.unsqueeze(1))
            generated_tokens = torch.cat(
                [generated_tokens, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, unfinished_sequences.unsqueeze(1)],
                dim=1
            )

            response_length += unfinished_sequences  # 更新单个响应的长度
            unfinished_sequences = unfinished_sequences & (
                next_token.squeeze(1) != tokenizer.eos_token_id)
            if not torch.any(unfinished_sequences):
                break

    output_ids = generated_tokens[:, prompt_length:]
    generated_text = tokenizer.decode(
        output_ids[0], skip_special_tokens=True)  # 解码单个响应

    return generated_text, response_length.item()  # 返回单个文本和其长度（item()用于从张量取值）


def generate_batch_fsdp(
    local_rank: int,
    model,
    tokenizer,
    prompt: str,  # 修改为单个字符串
    num_generations: int,  # 新增参数
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> Tuple[List[str], List[int]]:
    """
    为单个提示生成 num_generations 次响应，每次单独推理，以节省显存。
    """
    all_generated_texts: List[str] = []
    all_response_lengths: List[int] = []
    for i in range(num_generations):
        if local_rank == 0:
            print(
                f"Generating response {i+1}/{num_generations} for prompt: '{prompt[:50]}...'")

        # 调用辅助函数生成单个响应
        generated_text, response_length = _generate_single_response(
            local_rank=local_rank,
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        all_generated_texts.append(generated_text)
        all_response_lengths.append(response_length)
    return all_generated_texts, all_response_lengths


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
