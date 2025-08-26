accelerate launch grpo.py \
    --model_path "/data/models/qwen2.5-72B" \
    --epoch 5 \
    --batch_size 4 \
    --lr 2e-4 \
    --lora_r 16 \
    --output_dir "llama2-7b-qlora-deepspeed-output"