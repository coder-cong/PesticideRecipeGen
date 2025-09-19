# manual_download_dataset.py
import os
from datasets import load_dataset

def download_dataset_manually():
    """手动下载数据集到指定位置"""
    cache_dir = "/data/lyl/projs/PesticideRecipeGen/cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # 设置环境变量
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    
    print("开始下载数据集...")
    try:
        ds = load_dataset(
            "swulling/gsm8k_chinese",
            cache_dir=cache_dir
        )
        
        # 保存到本地
        local_path = "/data/lyl/projs/PesticideRecipeGen/train/GRPO/trlGrpo/data/gsm8k_chinese"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        ds.save_to_disk(local_path)
        
        print(f"✅ 数据集下载完成，保存到: {local_path}")
        print(f"训练集大小: {len(ds['train'])}")
        print(f"测试集大小: {len(ds['test'])}")
        
        return local_path
        
    except Exception as e:
        print(f"❌ 数据集下载失败: {e}")
        return None

if __name__ == "__main__":
    download_dataset_manually()
