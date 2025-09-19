import requests
from typing import List, Dict, Any, Optional
import json
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
    print(probs_sort)

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


def generate_test():

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


def sample_test():
    probs = torch.Tensor([0.2, 0.3, 0.1, 0.2, 0.1, 0.1]).unsqueeze(0)
    print(probs.shape)
    next_token = _sample_top_p(probs, 0.6)
    print(next_token)


def reward_test():
    from fsdp_qlora_grpo import compute_rewards
    with open("/root/projs/PesticideRecipeGen/data/distill/distill_data_alpaca.json", "r", encoding="utf-8") as f:
        import json
        data = json.load(f)
    rewards = compute_rewards(data[0]["instruction"], [data[0]["output"]])
    print(rewards)


def compute_logprobs_test():
    tokenizer = AutoTokenizer.from_pretrained("/data/models/qwen2.5-0.5B")
    tokens = tokenizer.encode("nihao", return_tensors="pt")
    print([tokenizer.decode(token) for token in tokens[0]])


def save_lora_weight():
    import os
    from pathlib import Path
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    # --- 配置 ---
    # 请将此替换为你要加载的7B模型名称或本地路径
    # 例如："mistralai/Mistral-7B-v0.1" 或 "Qwen/Qwen1.5-7B-Chat"
    # 注意：Llama系列模型可能需要Hugging Face token或本地下载
    MODEL_NAME = "/data/lyl/models/qwen2.5-7B"

    # 保存未训练LoRA适配器的目录
    OUTPUT_LORA_DIR = "./untrained_7b_lora"

    # --- 1. 加载基础模型和分词器 ---
    print(f"Loading base model: {MODEL_NAME}...")

    # 检查是否有可用的CUDA设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 建议使用4位量化加载基础模型以节省GPU内存
    # 如果内存充足，可以移除 quantization_config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,  # 启用4位量化
        torch_dtype=torch.bfloat16,      # 使用bfloat16进行计算
        device_map="auto",               # 自动将模型加载到可用设备
        trust_remote_code=True,          # 某些模型（如Qwen）需要此项
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True)

    # 某些模型可能没有pad_token，设置eos_token作为pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Base model and tokenizer loaded.")

    # --- 2. 定义LoRA配置 ---
    # 这些是 Mistral-7B-v0.1 通常的 LoRA 目标模块。
    # 对于其他模型，你可能需要根据其架构进行调整。
    # 常见的是 QKV 投影层，以及 MLP 层。
    lora_target_modules = ["q_proj", "v_proj", "k_proj",
                           "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=8,                           # LoRA的秩，决定了适配器的大小和表达能力
        lora_alpha=16,                 # LoRA缩放因子
        target_modules=lora_target_modules,  # 要应用LoRA的模块名称列表
        lora_dropout=0.05,             # LoRA层的dropout率
        bias="none",                   # 偏置参数的处理方式
        task_type=TaskType.CAUSAL_LM,  # 任务类型，这里是因果语言模型
    )
    print("LoRA configuration defined.")

    # --- 3. 创建 PEFT 模型（应用LoRA） ---
    # 这一步将LoRA层添加到基础模型上，但LoRA权重是未训练的。
    peft_model = get_peft_model(model, lora_config)

    # 打印模型中可训练参数的数量，主要是LoRA参数
    peft_model.print_trainable_parameters()
    print("PEFT (LoRA) model created (untrained).")

    # --- 4. 保存未训练的LoRA适配器 ---
    Path(OUTPUT_LORA_DIR).mkdir(parents=True, exist_ok=True)  # 创建输出目录
    peft_model.save_pretrained(OUTPUT_LORA_DIR)

    print(
        f"\nUntrained LoRA adapter saved to: {os.path.abspath(OUTPUT_LORA_DIR)}")
    print(
        f"你可以将此目录 '{os.path.abspath(OUTPUT_LORA_DIR)}' 作为 vLLM 的 lora_path 进行加载。")


def get_vllm_inference(
    prompt: str,
    num_responses: int = 1,
    vllm_server_url: str = "http://localhost:8000",
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    stop_sequences: Optional[List[str]] = None,
) -> List[str]:
    """
    通过请求本地部署的vLLM服务（/v1/completions API）来获取文本生成结果。
    此函数假设输入的prompt已经应用了chat template或其他必要的格式。
    Args:
        prompt (str): 输入的文本提示 (prompt)，此字符串应已包含模型所需的聊天模板格式。
        num_responses (int): 需要生成的回答数量 (对应OpenAI API的 'n' 参数)。默认为1。
        vllm_server_url (str): 本地vLLM服务的URL地址。默认为 "http://localhost:8000"。
        max_tokens (int): 生成的最大token数量。默认为256。
        temperature (float): 控制生成文本的随机性。较高的值会使输出更随机，较低的值会使输出更确定。默认为0.7。
        top_p (float): 控制生成文本的多样性。只考虑累积概率达到top_p的token。默认为0.95。
        stop_sequences (List[str], optional): 生成文本的停止序列列表。当模型生成到这些序列中的任何一个时，将停止生成。例如 ["\n", "###"]。默认为None。
    Returns:
        List[str]: 包含所有生成回答的字符串列表。如果请求失败或没有生成结果，则返回空列表。
    """
    generated_texts: List[str] = []

    api_endpoint = f"{vllm_server_url}/v1/completions"
    payload = {
        "prompt": prompt,
        "n": num_responses,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": False,  # We want the full response at once
    }
    if stop_sequences:
        payload["stop"] = stop_sequences
    print(f"Requesting vLLM text completion at: {api_endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2,ensure_ascii=False)}")
    try:
        response = requests.post(api_endpoint, json=payload, timeout=120)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        response_data = response.json()
        if 'choices' in response_data:
            for choice in response_data['choices']:
                # Text completion response structure: choice['text']
                if 'text' in choice:
                    generated_texts.append(choice['text'])
        else:
            print(f"Error: No 'choices' found in response: {response_data}")
    except requests.exceptions.Timeout:
        print(f"Error: Request to vLLM server timed out after 120 seconds.")
    except requests.exceptions.ConnectionError as e:
        print(
            f"Error: Could not connect to vLLM server at {vllm_server_url}. Is it running? Details: {e}")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON response from vLLM server.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return generated_texts


def show_model():
    with open("/data/lyl/model_structure.txt", "r", encoding="utf-8") as f:
        s = json.load(f)
        print(s)


if __name__ == "__main__":
    # sample_test()
    # reward_test()
    # compute_logprobs_test()
    # save_lora_weight()
    # from dataset import GRPODataset
    # tokenizer = AutoTokenizer.from_pretrained("/data/lyl/models/qwen2.5-7B")
    # data_path = "/data/lyl/projs/PesticideRecipeGen/data/distill/distill_data_alpaca.json"
    # dataset = GRPODataset(data_path, tokenizer)
    # print(get_vllm_inference(dataset[0], 8, max_tokens=1024))
    show_model()
