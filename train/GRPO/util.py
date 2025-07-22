import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import os
import pandas as pd


class TokenizerUtil:
    def __init__(self, model_path='/home/iiap/大语言模型/Meta-Llama-3-8B-Instruct'):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

        # 确保所有必要的特殊标记都被设置
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = '<s>'
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = '</s>'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_path = model_path
        # 更新特殊标记的ID
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, sent, max_length=1024):
        # 编码输入文本，包括特殊标记
        encoded = [self.bos_token_id] + self.tokenizer.encode(sent, add_special_tokens=True, max_length=max_length,
                                                              truncation=True) + [self.eos_token_id]

        # 确保序列长度正确
        if len(encoded) < max_length:
            encoded = encoded + [self.pad_token_id] * \
                (max_length - len(encoded))
        else:
            encoded = encoded[:max_length - 1] + [self.eos_token_id]

        input_ids = torch.LongTensor(encoded)
        attention_mask = (input_ids != self.pad_token_id).long()

        return input_ids, attention_mask

    def easy_encode(self, sent):

        return self.tokenizer.encode(sent)

    def decode(self, input_ids):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        if self.eos_token_id in input_ids:
            end = input_ids.index(self.eos_token_id) + 1
            input_ids = input_ids[:end]

        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    def pad_to_left(self, input_ids):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        if self.eos_token_id in input_ids:
            end = input_ids.index(self.eos_token_id)
            input_ids[end] = self.pad_token_id
            input_ids = input_ids[end:] + input_ids[:end]

        input_ids = torch.LongTensor(input_ids)
        attention_mask = (input_ids != self.pad_token_id).long()

        return input_ids, attention_mask


def read_promote_text(file_path):
    """读取文本文件并返回其中的内容作为 promote 文本。

    Args:
      file_path: 文本文件的路径。

    Returns:
      文本文件的内容，作为字符串。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            promote_text = f.read()
        return promote_text
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。")
        return None


class read_gsmk8k(Dataset):
    """GSM8K Chinese数据集加载器"""

    def __init__(self, data_dir, sys_prompt, split="train"):
        """
        初始化数据集

        参数:
            data_dir (str): 包含parquet文件的目录
            split (str): 数据分割，"train"或"test"
        """
        # 构建parquet文件路径
        file_path = os.path.join(data_dir, f"{split}-00000-of-00001.parquet")

        # 读取parquet文件
        self.data = pd.read_parquet(file_path)

        self.SYSTEM_PROMPT = sys_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """获取单个数据样本"""
        item = self.data.iloc[idx]

        # 获取中文问题，如果不存在则使用英文问题
        question = item.get('question_zh-cn', item.get('question', ''))

        # 获取参考答案
        answer_only = item.get('answer_only', '')
        if not answer_only and 'answer' in item:
            answer_lines = item['answer'].strip().split('\n')
            if answer_lines:
                answer_only = answer_lines[-1]

        # 创建完整提示
        prompt = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        return {
            "prompt": prompt,
            "question": question,
            "answer": answer_only
        }
# # 使用示例
# if __name__ == "__main__":
#     tokenizer_util = TokenizerUtil()
#
#     sample_text = "Hello, how are you today?"
#     input_ids, attention_mask = tokenizer_util.encode(sample_text)
#
#     print("Input IDs:", input_ids)
#     print("Attention Mask:", attention_mask)
#     print("Decoded:", tokenizer_util.decode(input_ids))
#
#     # 测试 pad_to_left
#     padded_ids, padded_mask = tokenizer_util.pad_to_left(input_ids)
#     print("Padded to left IDs:", padded_ids)
#     print("Padded to left Mask:", padded_mask)
