import json
import os
from uuid import uuid4

def convert_json_to_jsonl(input_files, output_file, id_prefix=''):
    # 打开输出文件，使用追加模式
    with open(output_file, 'a', encoding='utf-8') as out_f:
        for input_file in input_files:
            # 为每个文件生成一个唯一的标识符
            file_id = str(uuid4())[:8]

            # 读取JSON文件
            with open(input_file, 'r', encoding='utf-8') as in_f:
                data = json.load(in_f)

            for item in data:
                # 创建新的对象来存储转换后的数据
                new_item = {
                    "conversation_id": f"{id_prefix}{file_id}_{item['id']}",
                    "category": "Conversation",
                    "conversation": [],
                    "dataset": "TDT"
                }

                # 转换对话格式
                for conv in item["conversations"]:
                    if conv["from"] == "user":
                        new_item["conversation"].append({"human": conv["value"]})
                    elif conv["from"] == "assistant":
                        new_item["conversation"].append({"assistant": conv["value"]})

                # 将转换后的对象写入JSONL文件
                out_f.write(json.dumps(new_item, ensure_ascii=False) + '\n')


# 使用函数
input_files = ['/home/iiap/PycharmProjects/再次开始的deeplearning/util/jsons/fuck.json',
               '/home/iiap/PycharmProjects/再次开始的deeplearning/util/jsons/rule.json']  # 你可以添加任意数量的输入文件
output_file = '/home/iiap/PycharmProjects/再次开始的deeplearning/util/jsons/text.jsonl'

# 如果输出文件已存在，先删除它
if os.path.exists(output_file):
    os.remove(output_file)

convert_json_to_jsonl(input_files, output_file, id_prefix='TDT_')