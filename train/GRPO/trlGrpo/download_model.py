

from huggingface_hub import snapshot_download
import os

# 设置下载目录
model_dir = "/data/lyl/projs/PesticideRecipeGen/train/GRPO/trlGrpo/model"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
local_path = os.path.join(model_dir, "Qwen2.5-0.5B-Instruct")

# 确保目录存在
os.makedirs(model_dir, exist_ok=True)

print(f"开始下载模型: {model_name}")
print(f"保存路径: {local_path}")

try:
    snapshot_download(
        repo_id=model_name,
        local_dir=local_path,
        local_dir_use_symlinks=False,  # 不使用符号链接
        resume_download=True,  # 支持断点续传
    )
    print("✅ 模型下载完成!")
except Exception as e:
    print(f"❌ 下载失败: {e}")




