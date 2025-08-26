# 1. 确定你主机上模型存放的绝对路径
# 例如: /data/models
# 完整的模型路径就是 /data/models/Qwen1.5-72B-Chat
HOST_MODEL_PATH="/root/qwen2.5-72B"
# 2. 运行 Docker 容器
# -it: 交互式终端
# --rm: 容器退出后自动删除，保持系统干净
# --gpus all: 将所有可用的 NVIDIA GPU 挂载到容器中，这是必须的
# -v [主机路径]:[容器路径]: 这是数据卷挂载
#    我们将主机上的 ${HOST_MODEL_PATH} 挂载到容器的 /app/models
#    我们还将主机的 HF 缓存挂载到容器的 /root/.cache/huggingface
docker run -it  --gpus all \
  # --rm \
  --runtime=nvidia \
  -v ${HOST_MODEL_PATH}:/root/model \
  -v /root/projs:/root/projs \
  --entrypoint bash \
  vllm/vllm-openai 