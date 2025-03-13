#!/bin/bash

# 设置环境变量，指定使用的 GPU
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 运行分布式训练脚本
torchrun --nproc_per_node=4 --nnodes=1 --master_port=12355 sft.py \
    --data_path "/home/iiap/PycharmProjects/再次开始的deeplearning/data/sft_demo.jsonl" \
    --model_path "/home/iiap/大语言模型/Meta-Llama-3-8B-Instruct" \
    --output_dir "model" \
    --gradient_accumulation_steps 2 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --num_epochs 13 \
    --log_interval 30