import os
import argparse
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


if __name__ == "__main__":

    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/models/trained/qwen2.5-72B-sft")
    tokenizer.pad_token_id = tokenizer.eos_token_id  # TODO check if it exists first
    # Set up dataloader
    dataset = GRPODataset(
        "/root/projs/PesticideRecipeGen/data/distill/distill_data_alpaca.json", tokenizer)
    # # For distributed training, use DistributedSampler
    # sampler = DistributedSampler(dataset, seed=10)
    print(dataset[0])
    # Use the custom collate function in DataLoader
    dataloader = DataLoader(
        dataset, batch_size=1,)
    for batch in dataloader:
        print(batch)
        break
