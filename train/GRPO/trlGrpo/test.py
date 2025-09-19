from datasets import load_from_disk

# 加载数据集
local_path = "/data/lyl/projs/PesticideRecipeGen/train/GRPO/trlGrpo/data/gsm8k_chinese"
ds = load_from_disk(local_path)

# 正确的检查方法
print("=== 数据集信息 ===")
print(f"数据集类型: {type(ds)}")
print(f"数据集结构: {ds}")

# 检查训练集
if 'train' in ds:
    train_data = ds['train']
    print(f"\n=== 训练集信息 ===")
    print(f"样本数量: {len(train_data)}")
    print(f"字段名称: {train_data.column_names}")
    
    # 查看第一个样本
    print(f"\n=== 第一个样本 ===")
    sample = train_data[0]
    for key, value in sample.items():
        print(f"{key}: {str(value)[:200]}...")

# 检查测试集
if 'test' in ds:
    test_data = ds['test']
    print(f"\n=== 测试集信息 ===")
    print(f"样本数量: {len(test_data)}")
    print(f"字段名称: {test_data.column_names}")
