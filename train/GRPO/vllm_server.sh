export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python vllm_server.py \
      --model="/data/lyl/models/qwen2.5-7B" \
      --host="0.0.0.0" \
      --port=8000 \
      --gpu_memory_utilization=0.6 \
      > vllm_server.log &