import torch
from datasets import load_dataset
from transformers import default_data_collator, AutoModelForCausalLM, get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
from utils.util import TokenizerUtil
import os
import argparse
import random
import numpy as np
from torch.nn import CrossEntropyLoss
from util.get_dataset import read_jsonlines_dpo

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)





def print_gpu_info():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name()}")



def get_train_param(model_actor):
    return model_actor.parameters()


'''
最重要的函数！！！
loss function
'''


def calculate_loss(logits, labels, vocab_size):
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        # Calculate the loss
        loss = loss_fct(shift_logits, shift_labels)
    else:
        loss = None

    return loss


def main(args):

    # 设置随机种子
    seed = args.seed if args.seed is not None else 42
    
    set_seed(args.seed)
    print(f"Using seed: {args.seed}")
    print_gpu_info()

    tokenizer = TokenizerUtil(model_path=args.model_path)


    dataset = read_jsonlines_dpo(
        json_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_lens
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    model_actor = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True,
                                                       torch_dtype=torch.bfloat16, max_length=args.max_lens)

    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1
        )
        model_actor = get_peft_model(model_actor, peft_config)
        model_actor.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model_actor.parameters(), lr=args.learning_rate)

    scheduler = get_scheduler(
        "cosine_with_restarts",
        optimizer=optimizer,
        num_warmup_steps=int(0.05 * len(loader) * args.num_epochs),
        num_training_steps=len(loader) * args.num_epochs,

    )

    model_actor = model_actor.to(device)

    model_actor.train()

    epoch_losses = []
    nan_counts = []

    for epoch in range(args.num_epochs):

        epoch_loss = 0.0
        batch_count = 0
        nan_count = 0

        for i, data in enumerate(loader):

            input_ids = data['input_ids']
            attention_mask = data["attention_mask"]
            labels = data["labels"]


            input_ids=input_ids.to(device)

            out = model_actor(input_ids=input_ids, attention_mask=attention_mask)
            loss = calculate_loss(out.logits, labels, model_actor.config.vocab_size)

            if not torch.isnan(loss):

                loss.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 累计有效的loss和batch数
                epoch_loss += loss.item()
                batch_count += 1
            else:
                nan_count += 1
                print(f"Warning: NaN loss encountered in Epoch {epoch + 1}, Step {i + 1}")
                optimizer.zero_grad()
                continue

            if (i + 1) % args.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}, Step {i + 1}/{len(loader)}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


        # 计算全局平均loss（如果所有batch都是nan，则设为nan）
        global_avg_loss = epoch_loss / batch_count if batch_count > 0 else float('nan')
        epoch_losses.append(global_avg_loss)
        nan_counts.append(nan_count)



        print(f"Epoch {epoch + 1} completed.")
        print(f"Global Average Loss: {global_avg_loss:.4f}")
        print(f"NaN count: {nan_count}")
        print(f"Epoch Losses: {[f'{loss:.4f}' for loss in epoch_losses]}")
        print(f"NaN Counts: {nan_counts}")


    ##当前结束了一个epoch
    print("\nTraining completed.")
    print(f"Final global average loss: {epoch_losses[-1]:.4f}")
    valid_losses = [loss for loss in epoch_losses if not np.isnan(loss)]
    if valid_losses:
        best_loss = min(valid_losses)
        best_epoch = epoch_losses.index(best_loss) + 1
        print(f"Best global average loss: {best_loss:.4f} (Epoch {best_epoch})")
    else:
        print("Warning: All epochs resulted in NaN losses.")
    print(f"Total epochs: {args.num_epochs}")
    print(f"Total NaN counts: {sum(nan_counts)}")

    if args.use_lora:

        model_actor.save_pretrained(os.path.join(args.output_dir, 'peft_model'))

    else:
        # 如果不使用LoRA，直接保存完整模型
        model_actor.save_pretrained(os.path.join(args.output_dir, 'full_model'))
        tokenizer.tokenizer.save_pretrained(os.path.join(args.output_dir, 'full_model'))

    print("Training finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a causal language model with LoRA")
    parser.add_argument("--data_path", type=str,
                        default="/home/iiap/PycharmProjects/再次开始的deeplearning/data/sft_demo.jsonl",
                        help="Path to the dataset")
    parser.add_argument("--model_path", type=str, default="/media/iiap/25df545d-3a24-4466-b58d-f96c46b9a3bf/LargeModel/Qwen2.5-0.5B-Instruct",
                        help="Path to the pre-trained model")
    parser.add_argument("--output_dir", type=str, default="/media/iiap/25df545d-3a24-4466-b58d-f96c46b9a3bf/LargeModel/trained", help="Directory to save the trained model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval in steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=10721, help="Random seed for initialization")
    parser.add_argument("--max_lens", type=int, default=2048, help="训练模型单词的语句统一长度")
    parser.add_argument("--use_lora", type=bool, default=True, help="是否使用lora进行训练")
    parser.add_argument("--lora_rank", type=int, default=32, help="lora的质数")

    args = parser.parse_args()
    main(args)