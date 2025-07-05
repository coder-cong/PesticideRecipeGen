import json


def add_prefix_to_id_and_replace_keywords(file_path, id_prefix="tdtfuck", replacements=None):
    if replacements is None:
        replacements = {
            "user": "human",
            "assistant": "gpt"
        }

    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 遍历每个项目并修改ID和关键词
    for item in data:
        # 修改ID
        item['id'] = f"{id_prefix}{item['id']}"

        # 修改conversations中的关键词
        for conversation in item.get('conversations', []):
            if 'from' in conversation and conversation['from'] in replacements:
                conversation['from'] = replacements[conversation['from']]

    # 写回修改后的JSON文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 指定你的JSON文件路径
    json_file_path = '/home/iiap/下载/tdt_fuck.json'

    # 调用函数给ID添加前缀并替换关键词
    add_prefix_to_id_and_replace_keywords(json_file_path)