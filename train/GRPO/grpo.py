import os
import argparse
import torch as torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM,  BitsAndBytesConfig
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataset import GRPODataset


def load_dataset_test():
    data_path = "/root/projs/PesticideRecipeGen/data/distill/distill_data_alpaca.json"
    model_path = "/data/models/qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = GRPODataset(data_path=data_path, tokenizer=tokenizer)
    for data in dataset:
        print(data)


def init_ddp():
    dist.init_process_group("nccl")
    return (int(os.environ['RANK']), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GRPO训练")
    # 训练参数
    parser.add_argument("--epoch", type=int, help="训练轮次")
    parser.add_argument("--batch_size", type=int, help="批次大小")
    parser.add_argument("--log_iter", type=int, default=100, help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--output_dir", type=str,
                        default="/data/train/grpo", help="学习率")
    # 模型参数
    parser.add_argument("--model_path", type=str,
                        default="/data/models/qwen2.5-7B", help="模型路径")
    parser.add_argument("--data_path", type=str,
                        default="/root/projs/PesticideRecipeGen/data/distill/distill_data_alpaca.json", help="数据路径")
    ### NEW: LoRA specific arguments ###
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension (rank).")
    parser.add_argument("--lora_alpha", type=int,
                        default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float,
                        default=0.05, help="LoRA dropout.")

    args, _ = parser.parse_known_args()
    # 初始化Accelerator
    accelerator = Accelerator(log_with="wandb", project_dir="/data/train/grpo")
    accelerator.print(
        f"Using {accelerator.num_processes} GPUs with DeepSpeed and QLoRA.")

    # 加载分词器，准备数据集
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = GRPODataset(data_path=args.data_path, tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size)

    # 配置量化
    # 配置4-bit量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # 使用bfloat16进行计算
        bnb_4bit_use_double_quant=True,
    )

    # 加载量化后的模型
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 quantization_config=bnb_config,
                                                 # 将模型加载到当前进程对应的GPU上
                                                 device_map={"": accelerator.process_index}, )

    # 配置lora
    # 预处理模型以适配k-bit训练
    model = prepare_model_for_kbit_training(model)

    # 找到所有线性层以应用LoRA
    # 这部分可能需要根据具体模型进行调整
    # 打印model结构 (print(model)) 来找到目标模块
    target_modules = ["q_proj", "k_proj", "v_proj",
                      "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # 将LoRA适配器应用到模型上
    model = get_peft_model(model, lora_config)

    # 打印可训练参数的比例，你会看到这个数字非常小
    model.print_trainable_parameters()

    # --- 6. ### MODIFIED: 创建优化器和调度器 ### ---
    # DeepSpeed会自己处理优化器和调度器，我们只需要创建它们，Accelerate会正确地处理
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps)
    # --- 7. 核心步骤: 调用 accelerator.prepare() ---
    # Accelerate会用DeepSpeed引擎封装所有东西
    model, optimizer, train_dataloader,  lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    print(accelerator.num_processes)
