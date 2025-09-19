# test_compatibility_fixed.py  grpo环境配置测试
import torch
import transformers
import trl
import peft
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

def test_grpo_compatibility():
    print("=== GRPO兼容性测试 ===")
    
    try:
        # 修复配置参数
        config = GRPOConfig(
            output_dir="./test_output",
            learning_rate=5e-6,
            per_device_train_batch_size=1,
            num_generations=2,  # 生成数量
            generation_batch_size=2,  # 确保能被num_generations整除
            max_prompt_length=256,
            max_completion_length=128,
        )
        print("✓ GRPOConfig 创建成功")
        
        # 测试其他配置组合
        test_configs = [
            {"num_generations": 2, "generation_batch_size": 2},
            {"num_generations": 4, "generation_batch_size": 4},
            {"num_generations": 2, "generation_batch_size": 4},  # 4能被2整除
        ]
        
        for i, test_config in enumerate(test_configs):
            config_test = GRPOConfig(
                output_dir="./test_output",
                learning_rate=5e-6,
                per_device_train_batch_size=1,
                **test_config
            )
            print(f"✓ 测试配置 {i+1}: num_generations={test_config['num_generations']}, generation_batch_size={test_config['generation_batch_size']}")
        
        print("✓ 所有配置测试通过")
        
    except Exception as e:
        print(f"✗ GRPO测试失败: {e}")
        return False
    
    return True

def show_grpo_config_rules():
    print("\n=== GRPO配置规则 ===")
    print("1. generation_batch_size 必须能被 num_generations 整除")
    print("2. 推荐配置组合:")
    print("   - num_generations=1, generation_batch_size=1 (最节省显存)")
    print("   - num_generations=2, generation_batch_size=2")
    print("   - num_generations=4, generation_batch_size=4")
    print("   - num_generations=2, generation_batch_size=4 (批次更大)")
    print("   - num_generations=4, generation_batch_size=8")
    
    print("\n3. 显存使用估算:")
    print("   实际批次大小 = per_device_train_batch_size × gradient_accumulation_steps × generation_batch_size")
    print("   总生成数量 = 实际批次大小 × num_generations")

if __name__ == "__main__":
    show_grpo_config_rules()
    success = test_grpo_compatibility()
    if success:
        print("\n🎉 环境配置完全符合要求！")
    else:
        print("\n❌ 需要调整环境配置")
