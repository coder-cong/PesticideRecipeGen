import copy
import functools
import gc
import math
import os
import sys
import time
import types
from contextlib import nullcontext
from glob import glob
from pathlib import Path
from typing import Dict, List

import bitsandbytes as bnb
import safetensors
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from accelerate import init_empty_weights
from accelerate.utils import set_seed

# Model loading
from bitsandbytes.nn import Linear4bit, Params4bit
from fastcore.parallel import parallel

# Argument parsing
from fastcore.script import Param, bool_arg, call_parse
from packaging.version import parse
from safetensors.torch import save_file

# Torch + distributed training
from torch import Tensor, nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
    offload_wrapper,
)

# FSDP
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, hub
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2Attention, Qwen2MLP
import wandb
from profiling_utils import profiling_context

from dataset import GRPODataset

# DATASET + DATALOADERS (modified from llama recipes)
# Formatting prompts in alpaca
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
# Dataset class


class InstructionDataset(Dataset):
    def __init__(self, dataset, tokenizer, style="alpaca"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.style = style

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        if self.style == "guanaco":
            prompt = self.dataset[index]["text"].split("### Assistant: ")[0]
            example = self.dataset[index]["text"]
        elif self.style == "qna":
            prompt_template = "###Context:\n{context}\n###Question:\n{question}\n###Answer:\n"
            sample = self.dataset[index]
            prompt = prompt_template.format_map(sample)
            example = prompt + sample['answer']
        elif self.style == "qna_no_ctx":
            prompt_template = "###Question:\n{question}\n###Answer:\n"
            sample = self.dataset[index]
            prompt = prompt_template.format_map(sample)
            example = prompt + sample['answer']
        else:  # Alpaca
            ann = self.dataset[index]
            if ann.get("input", "") == "":
                prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
            else:
                prompt = PROMPT_DICT["prompt_input"].format_map(ann)
            example = prompt + ann["output"]

        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": example_mask.tolist(),
        }


class Logger:
    def __init__(self, args, log_to="stdout", project_name="fsdp_qlora", entity=None, group=None, name=None, rank=0):
        # self.log_every_n_steps = log_every_n_steps TODO: add this back as an option
        self.log_to = log_to
        if self.log_to == "wandb" and rank == 0:
            import wandb
            wandb.init(project=project_name, entity=entity,
                       group=group, name=name, config=args)

    def log(self, d: Dict, rank: int):
        if rank != 0:
            return
        if self.log_to == "tqdm":
            for k, v in d.items():
                tqdm.write(f'{k}: {v}')
        elif self.log_to == "wandb":
            wandb.log(d)
        elif self.log_to == "stdout":
            for k, v in d.items():
                print(f'{k}: {v}')

    def finish(self, rank=0):
        if self.log_to == "wandb" and rank == 0:
            wandb.finish()

# Main function, run on each process


def fsdp_main(local_rank: int, world_size: int, args: Dict):

    # Setup and initialize the process group
    os.environ['MASTER_ADDR'] = args["master_addr"]
    os.environ['MASTER_PORT'] = args["master_port"]

    rank = local_rank

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    # CPU卸载设置 将CPU核心平均分配给每个GPU进程
    torch.set_num_threads(
        os.cpu_count()//(min(world_size, torch.cuda.device_count())))

    # Start logging
    logger = Logger(args, log_to=args["log_to"], project_name=args["project_name"],
                    entity=args["entity"], group=args["group"], name=args["name"], rank=rank)
    # Timing stuff
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # model precision, qlora compute precison, and FSDP mixed precision policy.
    # The Linear4Bit quant_storage dtype should always match the FSDP param_dtype. The compute_dtype should match the AMP compute dtype.
    # MixedPrecision(param_dtype=fp32, reduce_dtype=fp32, buffer_dtype=fp32) uses `torch.amp.autocast` to control precision.
    # limited qlora testing shows that fp16 only works with autocast while bf16 trains with both pure and autocast modes.
    # TODO: test how often this holds for mp_fp16
    mp_policy = None
    load_param_skip_names = []
    if args["precision"] == "bf16":
        torch_dtype, compute_dtype = torch.bfloat16, torch.bfloat16
    elif args["precision"] == "fp32":
        torch_dtype, compute_dtype = torch.float32, torch.float16
    elif args["precision"] == "fp16_autocast":
        compute_dtype, torch_dtype = torch.float16, torch.float32
        mp_policy = MixedPrecision(
            param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    elif args["precision"] == "bf16_autocast":
        compute_dtype, torch_dtype = torch.bfloat16, torch.float32
        mp_policy = MixedPrecision(
            param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    elif args["precision"] == "bf16_buffers_autocast":
        compute_dtype, torch_dtype = torch.bfloat16, torch.bfloat16
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.float32)
        load_param_skip_names = ['inv_freq']
    else:
        raise ValueError("Invalid precision")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    tokenizer.pad_token_id = tokenizer.eos_token_id  # TODO check if it exists first

    # Set up dataloader
    dataset = GRPODataset(args["data_path"], tokenizer)
    # For distributed training, use DistributedSampler
    sampler = DistributedSampler(dataset, seed=args["seed"])

    # Use the custom collate function in DataLoader
    dataloader = DataLoader(
        dataset, batch_size=args["batch_size"],  sampler=sampler)
    if rank == 0:
        print("tokenizer加载完成")
    # TODO 测试通过添加读取数据集进行训练的代码

    # Create model
    cfg = None
    attn_impl = "sdpa"  # torch 2.2 sdpa uses flash attn 2
    if rank == 0 or args['verbose']:
        print("Creating model", rank)
    # 非量化微调
    if args["train_type"] in ["full", "lora", "custom_lora"]:
        if (args["low_memory"] and rank == 0) or (not args["low_memory"]):
            model = AutoModelForCausalLM.from_pretrained(
                args["model_name"],
                use_cache=False,
                torch_dtype=torch_dtype,
                _attn_implementation=attn_impl
            )
            dtype = torch_dtype if args["precision"] == "bf16" else None
            model.to(dtype=dtype, device="cpu" if args["low_memory"] else rank)

        else:
            cfg = AutoConfig.from_pretrained(args["model_name"])
            cfg.use_cache = False
            cfg._attn_implementation = attn_impl
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(
                    cfg, torch_dtype=torch_dtype)

            if args["precision"] == "bf16":
                model.to(torch_dtype)
    # 量化微调
    elif args["train_type"] in ["qlora", "custom_qlora", "hqq_lora", "hqq_dora", "bnb_dora", "bnb_llama_pro", "hqq_llama_pro"]:  # Our custom loading
        cfg = AutoConfig.from_pretrained(args["model_name"])
        cfg.use_cache = False
        cfg._attn_implementation = attn_impl
        skip_modules = ["lm_head"]

        if args["train_type"] in ["bnb_llama_pro", "hqq_llama_pro"]:
            llama_pro_path = Path(args["llama_pro_path"])
            num_original_layers, num_expanded_layers = llama_pro_path.name.split(
                "blk_exp-")[1].split("-")
            num_original_layers, num_expanded_layers = int(
                num_original_layers), int(num_expanded_layers)
            total_new_layers = num_expanded_layers - num_original_layers
            split = int(num_original_layers /
                        (num_expanded_layers - num_original_layers))
            new_layer_ids = [
                split+(split+1)*n for n in range(total_new_layers)]
            new_layer_names = [f"layers.{i}" for i in new_layer_ids]
            skip_modules += [str(lid) for lid in new_layer_ids]
            cfg.num_hidden_layers = num_expanded_layers

        # load model on meta device without calling init and replace nn.Linear with Linear4bit
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(cfg)
            model.model = replace_linear(model.model, Linear4bit, compute_dtype=compute_dtype,
                                         quant_type='nf4', quant_storage=torch_dtype, skip_modules=skip_modules)

            if rank == 0:
                print("空模型创建完成")

        model.is_loaded_in_4bit = True

        # Grab the safetensors files that hold the weights
        if args["train_type"] in ["bnb_llama_pro", "hqq_llama_pro"]:
            files = glob(str(llama_pro_path/"*.safetensors"))
        else:
            try:
                idx = hub.cached_file(
                    args["model_name"], SAFE_WEIGHTS_INDEX_NAME)
                files, _ = hub.get_checkpoint_shard_files(
                    args["model_name"], idx)
            except OSError:
                try:
                    # This means the model doesn't have a model.safetensors.index.json because it is not sharded
                    files = []
                    files.append(hub.cached_file(
                        args["model_name"], SAFE_WEIGHTS_NAME))
                except OSError as e:
                    # This means the model probably doesn't have a safetensors file
                    raise e

        # Load in the weights, using our custom load_and_quantize method which quantizes Params4bit on the fly
        # and then places each layer on CPU or meta if using low_memory to minimize GPU memory usage
        def load_and_quantize_parallel(name_param, model, **kwargs):
            name, param = name_param
            load_and_quantize(rank, model, name, param, **kwargs)

        quant_method = "hqq" if args["train_type"] in [
            "hqq_lora", "hqq_dora", "hqq_llama_pro"] else "bnb"
        param_count = sum((p.numel() for n, p in model.named_parameters()))
        if rank == 0 or args['verbose']:
            print("Loading model", rank)
        if rank == 0 and args['verbose']:
            print(f"Total model params: {param_count}")

        # TODO 研究n_workers计算逻辑
        n_workers = n_loading_workers(
            quant_method, param_count) if args["loading_workers"] == -1 else args["loading_workers"]
        if rank == 0 and args['verbose']:
            print(f"Using n_workers: {n_workers} for loading")

        start = time.time()
        for filename in tqdm(files, desc="Loading & Quantizing Model Shards", disable=rank != 0, position=0):
            weights = safetensors.torch.load_file(filename)
            # TODO 可能需要修改移动设备的逻辑
            parallel(load_and_quantize_parallel, iter(weights.items()), n_workers=n_workers, threadpool=True,
                     model=model, dtype=torch_dtype, device=local_rank, skip_names=load_param_skip_names,
                     to_cpu=(args["low_memory"] and rank == 0), to_meta=(args["low_memory"] and rank != 0),
                     verbose=args["verbose"], quant_method=quant_method, is_dora=(args["train_type"] in ["hqq_dora", "bnb_dora"]))

        if rank == 0 and args["verbose"]:
            print(f"Loaded model weights in {time.time()-start:.3f} seconds")
        # cleanup any extra memory usage from parallel loading
        torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0 or args['verbose']:
        print(
            f"Rank {rank}: Model created: {torch.cuda.memory_reserved(local_rank)/2**30:.3f} GiB")
        print("模型量化完成，准备添加LoRA层")
    # PEFT setup (LoRA and QLoRA)
    if args["train_type"] in ["lora", "qlora"]:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            lora_alpha=args["lora_alpha"],
            lora_dropout=args["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj",
                            "o_proj", "up_proj", "down_proj"]
        )
        # PEFT will move quant_state to meta device, so this method prevents that
        # from happening by replacing quant_state.to with a dummy function
        if rank != 0 and args["low_memory"]:
            setup_quantized_meta_for_peft(model)

        model = get_peft_model(model, peft_config)

        if rank == 0:
            model.print_trainable_parameters()
        elif args['low_memory']:
            # And then setup_quantized_peft_meta_for_training sets quant_state.to back to normal
            setup_quantized_peft_meta_for_training(model)
    if rank == 0:
        print("LoRA Adapter添加完成")

    # Wrap model with llama-recipies or custom LoRA policy
    my_auto_wrap_policy = get_wrapping_policy(custom_policy=args["train_type"] in ["custom_qlora", "hqq_lora", "hqq_dora", "bnb_dora"],
                                              vanilla_policy=args["train_type"] in ["full", "bnb_llama_pro", "hqq_llama_pro"])

    if rank == 0 or args['verbose']:
        print("Wrapping model w/ FSDP", rank)

    if args["sharding_strategy"] == "full_shard":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif args["sharding_strategy"] == "shard_grad_op":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif args["sharding_strategy"] == "ddp":
        sharding_strategy = ShardingStrategy.NO_SHARD
    elif args["sharding_strategy"] == "hybrid_full_shard":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif args["sharding_strategy"] == "hybrid_shard_grad_op":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        raise ValueError("Invalid FSDP sharding strategy")

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=my_auto_wrap_policy,
        # backward_prefetch=None, #BackwardPrefetch.BACKWARD_PRE
        use_orig_params=False,
        cpu_offload=CPUOffload(
            offload_params=True) if args["use_cpu_offload"] else None,
        limit_all_gathers=True,  # See https://github.com/pytorch/pytorch/issues/91165
        device_id=torch.cuda.current_device(),
        sync_module_states=args["low_memory"],
        param_init_fn=lambda module: module.to_empty(
            device=torch.device("cuda"), recurse=False)
        # TODO note about meta device and why we need this
        if (rank != 0 and args["low_memory"]) else None,
        mixed_precision=mp_policy,
    )
    if rank == 0 or args['verbose']:
        print(
            f"Rank {rank}: Wrapped model: {torch.cuda.memory_reserved(local_rank)/2**30:.3f} GiB")
    if args["log_to"] == 'wandb':
        logger.log(
            {"memory/allocated_after_model_wrap": torch.cuda.memory_allocated(local_rank)}, rank)
        logger.log(
            {"memory/reserved_after_model_wrap": torch.cuda.memory_reserved(local_rank)}, rank)

    # Synchronize at the start
    dist.barrier()

    # Apply activation checkpointing
    if args["use_gradient_checkpointing"]:
        if args['reentrant_checkpointing']:
            model.enable_input_require_grads()
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.REENTRANT if args[
                'reentrant_checkpointing'] else CheckpointImpl.NO_REENTRANT,

        )

        def check_fn(submodule): return isinstance(
            submodule, (Qwen2DecoderLayer))
        if rank == 0 or args['verbose']:
            print("Applying activation checkpointing", rank)
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

    if args["use_activation_cpu_offload"]:
        if rank == 0 or args['verbose']:
            print("Applying activation offloading", rank)
        model = offload_wrapper(model)

    if rank == 0 and args['verbose']:
        print("Config:")
        print(cfg)
        print("Model:")
        print(model)
        print("Starting training")

    # Create the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], betas=(0.9, 0.95),
                                  eps=1e-5, weight_decay=args['wd'], fused=args["optimizer"] == "fused_adamw")
    if rank == 0:
        print("Optimizer 创建成功")
    # LR scheduler.
    gradient_accumulation_steps = max(1, args['gradient_accumulation_steps'])
    lr_scheduler, num_training_steps = get_lr_scheduler(
        optimizer, dataloader, gradient_accumulation_steps, args)

    # Sanity check: see what parameters the optimizer has and which require grad:
    if rank == 0 and args['verbose']:
        print("Optimizer params:")
        for group in optimizer.param_groups:
            for param in group['params']:
                pass
                # print(
                #     f"Shape: {param.shape}, Requires Grad: {param.requires_grad}, Dtype: {param.dtype}")

    # Autocast for mixed precision with fp16/bf16 compute types with fp32 params
    if args["precision"] in ["fp16_autocast", "bf16_autocast", "bf16_buffers_autocast"]:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=compute_dtype)
    else:
        autocast = nullcontext()
    scaler = ShardedGradScaler(
    ) if args["precision"] == "fp16_autocast" else None
    scale_grads = scaler is not None

    if rank == 0:
        print("Total Training Steps:", num_training_steps)
    memory_stats = []
    progress_bar = tqdm(range(num_training_steps), disable=rank != 0)
    init_start_event.record()
    log_loss, log_lr = 0.0, -1
    # Reset peak memory to track that
    torch.cuda.reset_peak_memory_stats(local_rank)
    with profiling_context(args, rank=rank) as prof:
        for epoch in range(args['num_epochs']):
            update_progress_bar(progress_bar, epoch, log_loss, log_lr, rank)
            model.train()
            ddp_loss = torch.zeros(2).to(local_rank)

            for batch_idx, batch in enumerate(dataloader):

                accumulate_grads = (
                    batch_idx+1) % gradient_accumulation_steps == 0

                # Prevent gradient syncing until update step if using no_sync option.
                # Documentation states this should only be used on the root FSDP instance
                # We assume this is a one-node setup
                if args['no_sync'] and not accumulate_grads:
                    sync_context = model.no_sync()
                else:
                    sync_context = nullcontext()

                # Start logging memory (first iter) if requested
                if args['profile_memory'] and batch_idx == 0 and rank == 0 and epoch == 0:
                    torch.cuda.memory._record_memory_history()

                # Log memory usage
                if batch_idx == 0 and epoch == 0 and (rank == 0 or args['verbose']):
                    reserved_before_forward = torch.cuda.memory_reserved(
                        local_rank)
                    memory_stats.append(
                        f"Rank {rank}: Before forward: {reserved_before_forward/2**30:.2f} GiB")
                    if args["log_to"] == 'wandb':
                        logger.log(
                            {"memory/allocated_before_forward": torch.cuda.memory_allocated(local_rank)}, rank)
                        logger.log(
                            {"memory/reserved_before_forward": reserved_before_forward}, rank)

                # 获取问题和答案
                prompt = batch[0]  # 取第一个元素，因为batch_size=1

                # Forward pass
                with sync_context:
                    with autocast:
                        # 1. 生成多个候选答案
                        responses, prompt_lens = generate_batch_fsdp(
                            rank,
                            model,
                            tokenizer,
                            [prompt]*args["num_generations"],
                            max_new_tokens=args["max_completion_length"],
                            temperature=args["temperature"]
                        )
                        # responses, prompt_length = generate_responses(
                        #     rank,
                        #     model,
                        #     tokenizer,
                        #     prompt,
                        #     num_generations=args["num_generations"],
                        #     max_new_tokens=args["max_completion_length"],
                        #     temperature=args["temperature"]
                        # )
                        if rank == 0:
                            print(responses)
                        # 2. 计算奖励得分

                        # 3. 计算policy和ref模型logprobs
                        # 4. 计算loss
                        # 5. 反向传播
                        output = model(
                            batch['input_ids'].to(local_rank),
                            labels=batch['labels'].to(local_rank),
                            attention_mask=None,
                        )
                        loss = output.loss

                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                    # Log memory usage
                    if batch_idx == 0 and epoch == 0 and (rank == 0 or args['verbose']):
                        reserved_after_forward = torch.cuda.memory_reserved(
                            local_rank)
                        memory_stats.append(
                            f"Rank {rank}: After forward: {reserved_after_forward/2**30:.2f} GiB")
                        if args["log_to"] == 'wandb':
                            logger.log(
                                {"memory/allocated_after_forward": torch.cuda.memory_allocated(local_rank)}, rank)
                            logger.log(
                                {"memory/reserved_after_forward": reserved_after_forward}, rank)

                    # Backward pass
                    if scale_grads:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # Record loss
                bs = batch['input_ids'].shape[0]
                ddp_loss[0] += loss.item() * bs * gradient_accumulation_steps
                ddp_loss[1] += bs

                # Step the optimizer (w/ gradient accumulation)
                if accumulate_grads:
                    if args['apply_gradient_clipping'] and (args['grad_norm'] is not None):
                        model.clip_grad_norm_(args['grad_norm'], norm_type=2.0)
                    if scale_grads:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    # avoid overhead when lr is constant.
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    progress_bar.update(1)

                # Log memory usage after backward
                if batch_idx == 0 and epoch == 0 and (rank == 0 or args['verbose']):
                    reserved_after_backward = torch.cuda.memory_reserved(
                        local_rank)
                    memory_stats.append(
                        f"Rank {rank}: After backward: {reserved_after_backward/2**30:.2f} GiB")
                    if args["log_to"] == 'wandb':
                        logger.log(
                            {"memory/allocated_after_backward": torch.cuda.memory_allocated(local_rank)}, rank)
                        logger.log(
                            {"memory/reserved_after_backward": reserved_after_backward}, rank)

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

    # Synchronize at the end and record time
    init_end_event.record()
    dist.barrier()
    torch.cuda.synchronize()

    if rank == 0:
        print("Finished training", rank)

    # Print time, model, & memory stats
    time_taken = init_start_event.elapsed_time(init_end_event) / 1000
    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        print(f"CUDA event elapsed time: {time_taken} sec")
        logger.log({"time_taken": time_taken}, rank)
    for line in memory_stats:
        print(line)

    # End logging
    logger.finish(rank=rank)

    # Save model - ref: https://github.com/pytorch/pytorch/issues/98823
    # HQQLinear custom state_dict() method causes issues when saving.
    # Model is saved fine when `state_dict()` method is removed.
    # Non param/buffer types are not saved with FSDP.
    # It might be better to just save the trained lora layers.
    # summon_full_params on lora layers and save.
    if args["save_model"]:
        if rank == 0:
            os.makedirs(args["output_dir"], exist_ok=True)
        dist.barrier()
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        if args["train_type"] in ["custom_lora", "custom_qlora", "hqq_lora", "hqq_dora", "bnb_dora", "bnb_llama_pro", "hqq_llama_pro"]:
            cpu_state_dict = {}
            if args["train_type"] in ["bnb_llama_pro", "hqq_llama_pro"]:
                trainable_fsdp_modules = [
                    (n, m) for n, m in model.named_modules() if n.endswith(tuple(new_layer_names))]
            else:
                trainable_fsdp_modules = [(n, m) for n, m in model.named_modules(
                ) if n.endswith(('lora_AB', 'dora_layer', 'magnitude_layer'))]
            for prefix, module in trainable_fsdp_modules:
                prefix = (prefix.replace("_fsdp_wrapped_module.", "")
                                .replace("_checkpoint_wrapped_module.", "")
                                .replace("_offload_wrapped_module.", ""))
                if args['verbose']:
                    print(f"Saving {prefix}")
                with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, save_policy):
                    cpu_state_dict = {
                        **cpu_state_dict, **{f"{prefix}.{k}": v for k, v in module.state_dict().items()}}
                dist.barrier()
                torch.cuda.synchronize()
            if rank == 0:
                print("Saving trained LoRA weights.")
                save_file(cpu_state_dict, os.path.join(
                    args["output_dir"], "model_state_dict.safetensors"))
                print("Done", rank)
        else:
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = model.state_dict()
                if rank == 0:
                    print("Saving full model weights.")
                    save_file(cpu_state_dict, os.path.join(
                        args["output_dir"], "model_state_dict.safetensors"))
                    print("Done", rank)

    dist.barrier()  # Stop other processes ending while model saving - probably not needed?

    # Clean up
    dist.destroy_process_group()

# Utilities related to model loading


def replace_linear(model: nn.Module, linear_replacement: nn.Module, quant_config: dict | None = None,
                   skip_modules: List[str] = ["lm_head"], **kwargs):
    """
    Replace linear modules with a new Linear module.
    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        linear_replacement (`torch.nn.Module`):
            The linear module that replaces the old one. Only expects standard arguments.
            If other arguments need to be passed, use a lambda.
        skip_modules (`List[str]`, *optional*, defaults to `lm_head`):
            List of modules names not to convert. Defaults to `lm_head`.
    """
    for name, module in model.named_children():
        if name in skip_modules:
            continue

        # 深度优先替换模型所有线性层为量化的层
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement,
                           quant_config, skip_modules, **kwargs)

        if isinstance(module, torch.nn.Linear):
            if issubclass(linear_replacement, Linear4bit):
                model._modules[name] = linear_replacement(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    **kwargs
                )
            # 其他量化方式
            # elif issubclass(linear_replacement, HQQLinear):
            #     model._modules[name] = linear_replacement(
            #         module, quant_config, **kwargs)
            else:
                raise ValueError(
                    f"Unsupported linear replacement: {type(linear_replacement)}")
    return model


def load_and_quantize(rank: int, module: nn.Module, name: str, value: Tensor, device: torch.device = None, dtype: torch.dtype = None,
                      skip_names: list[str] = [], to_cpu: bool = False, to_meta: bool = False, verbose: bool = False, quant_method: str = 'bnb',
                      is_dora: bool = False):
    # TODO 这里为了节省内存将rank不为0的进程量化之后的结果保存到了meta设备上，内存足够后续可以尝试保存到内存中
    """
    Loads `value` tensor into submodule of `module`, optionally skipping `skip_names` and converting to `dtype`.

    Quantizes `Params4bit` on `device` then places on "cpu" if to_cpu=True or "meta" if to_meta=True.
    """
    def place_on_device(value):
        if to_meta:
            device = 'meta'
        elif to_cpu:
            device = 'cpu'
        return value.to(device=device, dtype=dtype)

    if any([skip_name in name for skip_name in skip_names]):
        if verbose:
            print(f"Skipping {name} because it is in skip_names")
        return

    module_key, _, value_key = name.rpartition('.')
    try:
        submodule = module.get_submodule(module_key)
    except AttributeError as e:
        print(f"Module {module_key} not found:\n{e}")
        return

    try:
        if quant_method == 'bnb':
            param = submodule.get_parameter(value_key)
            # 如果是需要量化的权重，量化并移动到设备上，如果不需要量化直接移动到对应设备上
            # if rank == 0:
            #     print(
            #         f"type:{type(param)},shape:{param.shape},dtype:{dtype},value.type:{type(value)}value.shape:{value.shape}")
            if isinstance(param, Params4bit):
                # With `sync_module_states=True`, a meta device Params4bit needs to be the same
                # shape as the quantized Params4bit with an initialized quant_state. However,
                # FSDP only syncs parameters and buffers, so the quant_state isn't copied. This
                # workaround quantizes Params4bit to initialize quant_state on all ranks, then
                # replaces Params4bit's data with a meta tensor to free memory on non-rank 0.
                if is_dora:
                    setattr(submodule, "dora_scale", value.norm(
                        p=2, dim=1).to(dtype=dtype).to("cpu"))
                value = type(param)(
                    value.to(device=device, dtype=dtype).data, **param.__dict__).cuda(device)

                if to_meta:
                    value = type(param)(
                        value.data.to("meta"), **value.__dict__)
                elif to_cpu:
                    value = type(param)(value.data.to("cpu"), **value.__dict__)
            else:
                value = type(param)(place_on_device(value).data)

        # 其他量化方式
        # elif quant_method=='hqq':
        #     if isinstance(submodule, HQQLinear):
        #         if value_key == "weight":
        #             # Like `Params4bit`, this workaround quantizes `HQQLinear`` per device so the quantization
        #             # meta dictionary is created on all ranks, before converting to meta on non-rank 0.
        #             submodule.linear_layer.to_empty(device=device)
        #             submodule.linear_layer.weight.data.copy_(value.to(device=device, dtype=dtype))
        #             if is_dora:
        #                 setattr(submodule, "dora_scale", value.norm(p=2, dim=1).to(dtype=dtype).to("cpu"))
        #             submodule.initialize()

        #             if to_meta:
        #                 setattr(submodule, "W_q", nn.Parameter(submodule.W_q.to("meta")))
        #             elif to_cpu:
        #                 setattr(submodule, "W_q", nn.Parameter(submodule.W_q.to("cpu")))
        #             submodule.in_gpu = False

        #         if value_key == "bias":
        #             raise ValueError("Bias not supported in HQQLinear yet!")
        #     else:
        #         param = submodule.get_parameter(value_key)
        #         value = type(param)(place_on_device(value).data)

    except AttributeError:
        # it's a buffer
        value = place_on_device(value)
    # 量化之后的value替换到model中
    setattr(submodule, value_key, value)


def n_loading_workers(quant_method: str, param_count: float):
    devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
    left = int(os.cpu_count()/torch.cuda.device_count())
    right = int((4 if quant_method == "hqq" else 8) *
                (devprops.total_memory/1e9/40) * (70/(param_count/1e9)))
    return min(left, right)


def setup_quantized_meta_for_peft(model: nn.Module):
    """Replaces `quant_state.to` with a dummy function to prevent PEFT from moving `quant_state` to meta device"""

    def temp_to_method(self, *args, **kwargs):
        return self
    for param in model.parameters():
        if isinstance(param, Params4bit):
            param.quant_state._orig_to = param.quant_state.to
            param.quant_state.to = types.MethodType(
                temp_to_method, param.quant_state)


def setup_quantized_peft_meta_for_training(model: nn.Module):
    """Replaces dummy `quant_state.to` method with the original function to allow training to continue"""
    for param in model.parameters():
        if isinstance(param, Params4bit) and hasattr(param.quant_state, '_orig_to'):
            param.quant_state.to = param.quant_state._orig_to
            param.quant_state._orig_to = None

# Wrap the model using LoRA policy from llama-recipes or custom policy:
# This checks for lora layers (has weight and requires_grad)


def get_wrapping_policy(custom_policy: bool = False, vanilla_policy: bool = False):
    from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2Attention, Qwen2MLP

    # if custom_policy:
    #     def lambda_policy_fn(module):
    #         # LoRA and DoRA trainable layers.
    #         return (isinstance(module, nn.Sequential) and all(m.weight.requires_grad for m in module)) or (isinstance(module, (DORALayer, MagnitudeLayer)))
    # else:

    def lambda_policy_fn(module):
        # 针对lora适配器层
        return (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        )

    def self_attn_policy_fn(module):
        # 针对注意力层
        # Check module name is self_attn.
        return isinstance(module, Qwen2Attention)

    def mlp_policy_fn(module):
        # 针对MLP层
        # Check module name is self_attn.
        return isinstance(module, Qwen2MLP)

    lambda_policy = functools.partial(
        lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    self_attn_policy = functools.partial(
        lambda_auto_wrap_policy, lambda_fn=self_attn_policy_fn)
    mlp_policy = functools.partial(
        lambda_auto_wrap_policy, lambda_fn=mlp_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )
    if vanilla_policy:
        return transformer_wrap_policy

    # lambda_policy包裹lora层，transformer_wrap_policy包裹transformer层
    policies = [lambda_policy, transformer_wrap_policy]
    if custom_policy:
        policies.extend([self_attn_policy, mlp_policy])
    return functools.partial(_or_policy, policies=policies)


def get_lr_scheduler(optimizer: optim.Optimizer, dataloader: DataLoader, gradient_accumulation_steps: int, args: Dict):
    """Returns linear, cosine, or constant learning rate scheduler"""
    num_training_steps = args['num_epochs'] * \
        len(dataloader) // gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * 0.1)
    if args['lr_scheduler'] == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps)
    elif args['lr_scheduler'] == "cosine":
        lr_scheduler = get_cosine_one_cycle_scheduler(
            optimizer, num_warmup_steps, num_training_steps, min_lr_fraction=0.1)
    elif args['lr_scheduler'] == "constant":
        lr_scheduler = None
    else:
        raise NotImplementedError(
            f"{args['lr_scheduler']} LR scheduler not implemented yet")
    return lr_scheduler, num_training_steps


def get_cosine_one_cycle_scheduler(optimizer: optim.Optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_fraction: float = 0.1):
    "A more general cosine scheduler with to control the minimum learning rate"
    lr_lambda = functools.partial(
        _get_cosine_one_cycle_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_fraction=min_lr_fraction
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def _get_cosine_one_cycle_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, min_lr_fraction=0.1,
):

    # LR scheduler.
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    scale_term = (1 - min_lr_fraction)
    progress = float(current_step - num_warmup_steps) / \
        float(max(1, num_training_steps - num_warmup_steps))
    return (math.cos(math.pi * progress)+1) * 0.5 * scale_term + min_lr_fraction


def update_progress_bar(progress_bar: tqdm, epoch: int, log_loss: float, log_lr: float, rank: int):
    """Updates the progress bar with the current epoch, loss, and learning rate"""
    if rank == 0:
        if log_lr >= 0:
            progress_bar.set_description(
                f"Epoch {epoch}, Loss {log_loss:.3f}, LR {log_lr:.2e}", refresh=True)
        else:
            progress_bar.set_description(
                f"Epoch {epoch}, Loss {log_loss:.3f}", refresh=True)

# And to get the dataloader


def get_dataloader(tokenizer: PreTrainedTokenizerFast, args: Dict):
    """Creates a dataset and appropriate dataloader with distributed sampler."""
    # Importing here rather than at the start to avoid multiprocessing issues
    from datasets import Dataset, load_dataset

    # Load the source dataset
    if args["dataset"] == "alpaca":
        dataset = load_dataset("yahma/alpaca-cleaned")['train']
    elif args["dataset"] == "alpaca_sample":
        dataset = load_dataset("yahma/alpaca-cleaned",
                               split=f"train[:{args['dataset_samples']}]")
    elif args["dataset"] == "dummy":
        dataset = Dataset.from_dict({
            'instruction': ["instruction"]*args["dataset_samples"],
            'input': ["input"]*args["dataset_samples"],
            'output': ["output"*args["context_length"]*2]*args["dataset_samples"]}  # A long output to test memory usage (gets truncated)
        )
    elif args["dataset"] == "guanaco":
        dataset = load_dataset(
            "timdettmers/openassistant-guanaco", split="train")
    elif args["dataset"] == "sql":
        dataset = load_dataset("knowrohit07/know_sql")['validation']
        dataset = dataset.shuffle(seed=args["seed"])
        dataset = dataset.select(range(1000, len(dataset)))
    elif args["dataset"] == "orca_math":
        dataset = load_dataset(
            "microsoft/orca-math-word-problems-200k")['train'].shuffle(seed=42)
        # train with 10k for starters. Then 100k.
        dataset = dataset.select(range(0, args['dataset_samples']))

    # truncate dataset so it's evenly divisible by grad_accumulation_steps
    dataset = dataset.select(range(0, len(dataset)-len(dataset) %
                             (args["batch_size"]*args["gradient_accumulation_steps"])))

    # # Create the InstructionDataset
    if args["dataset"] == "guanaco":
        dataset = InstructionDataset(dataset, tokenizer, style="guanaco")
    elif args["dataset"] == "sql":
        dataset = InstructionDataset(dataset, tokenizer, style="qna")
    elif args["dataset"] == "orca_math":
        dataset = InstructionDataset(dataset, tokenizer, style="qna_no_ctx")
    else:  # (w/ alpaca prompt formatting)
        dataset = InstructionDataset(dataset, tokenizer, style="alpaca")

    # Collate function
    def collate_fn(batch, with_attention_mask=False):
        # To list of tensors
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_masks = [torch.tensor(
            item['attention_mask']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]
        # Pad + truncate
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)[
            :, :args["context_length"]]
        if with_attention_mask:
            attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)[
                :, :args["context_length"]]
        else:
            attention_masks = None
        labels = pad_sequence(labels, batch_first=True,
                              padding_value=-100)[:, :args["context_length"]]
        # Return dict
        return {'input_ids': input_ids, 'attention_mask': attention_masks, 'labels': labels}

    # For distributed training, use DistributedSampler
    sampler = DistributedSampler(dataset, seed=args["seed"])

    # Use the custom collate function in DataLoader
    dataloader = DataLoader(
        dataset, batch_size=args["batch_size"], collate_fn=collate_fn, sampler=sampler)

    return dataloader


def generate_responses(local_rank, model, tokenizer, prompt_tokens, num_generations=4, max_new_tokens=100, temperature=1.0):
    """生成多个候选响应"""
    model.eval()  # 设置为评估模式

    assert len(prompt_tokens.shape) == 1
    # 将输入移动到模型所在设备
    # input_ids = prompt_tokens.input_ids.to(device)
    input_ids = prompt_tokens.to(local_rank)
    # attention_mask = prompt_tokens.attention_mask.to(device)
    prompt_length = input_ids.shape[0]

    responses = []

    # 生成多个响应
    input_ids = input_ids.unsqueeze(0).repeat(num_generations, 1)
    assert input_ids.shape == (num_generations, prompt_length)
    with torch.no_grad():
        outputs = model(
            input_ids,
            # attention_mask=attention_mask,
            # max_new_tokens=max_new_tokens,
            # do_sample=True,
            # temperature=temperature,
            # top_p=0.9,
            # pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        if local_rank == 0:
            print(outputs.logits.shape)
        # 提取生成的文本（不包括提示部分）
        generated_ids = outputs[:, prompt_length:]
        generated_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)
        responses.append(generated_text)

    return responses, prompt_length


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
    local_rank,
    model,
    tokenizer,
    prompt: str,
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
    device = f"cuda:{local_rank}"
    # 1. 设置 Tokenizer 并进行批处理编码
    # **关键**: 对于自回归生成，必须使用左填充！
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048  # 设定一个合理的最大输入长度
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

    return generated_texts, response_lengths.cpu().tolist()


if __name__ == "__main__":
    world_size = 8
    import json
    with open("config.json", "r", encoding="utf-8") as f:
        args = json.load(f)

    try:
        # Run
        mp.spawn(fsdp_main,
                 args=(world_size, args),
                 nprocs=torch.cuda.device_count(),
                 join=True)
    except AttributeError:
        dist.destroy_process_group()
