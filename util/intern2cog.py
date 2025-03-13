import json
import os
import shutil
from pathlib import Path

def create_dataset(input_json_path, input_image_folder, output_folder):
    # 创建输出文件夹
    output_path = Path(output_folder)
    images_path = output_path / "images"
    labels_path = output_path / "labels"
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    # 读取输入JSON文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 用于生成唯一文件名的计数器
    counter = 1

    # 处理每个项目
    for item in data:
        image_path = Path(item['image'])
        conversations = item['conversations']

        # 确保conversations是成对的（问答对）
        for i in range(0, len(conversations), 2):
            if i + 1 < len(conversations):
                # 构建新的文件名
                new_filename = f"{counter:07d}"
                counter += 1

                # 复制并重命名图片
                new_image_path = images_path / f"{new_filename}{image_path.suffix}"
                shutil.copy(image_path, new_image_path)

                # 创建新的JSON文件，只包含一个问答对
                new_json_data = {
                    "conversations": [
                        {
                            "role": "user" if conversations[i]['from'] == "human" else "assistant",
                            "content": conversations[i]['value']
                        },
                        {
                            "role": "user" if conversations[i+1]['from'] == "human" else "assistant",
                            "content": conversations[i+1]['value']
                        }
                    ]
                }
                new_json_path = labels_path / f"{new_filename}.json"
                with open(new_json_path, 'w', encoding='utf-8') as f:
                    json.dump(new_json_data, f, ensure_ascii=False, indent=2)

    print(f"Dataset created in {output_folder}")

# 使用示例
input_json_path = "/home/iiap/PycharmProjects/再次开始的deeplearning/util/converted_data.json"
input_image_folder = "/home/iiap/桌面/数据集/VQA"
output_folder = "/home/iiap/桌面/数据集/cogdata"

create_dataset(input_json_path, input_image_folder, output_folder)