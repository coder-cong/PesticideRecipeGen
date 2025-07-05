import json

# 读取JSON文件
input_file = '/home/iiap/PycharmProjects/再次开始的deeplearning/util/temp.json'  # 替换为您的输入文件名
output_file = '/home/iiap/PycharmProjects/再次开始的deeplearning/util/test.jsonl'  # 替换为您想要的输出文件名

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历数据并删除 "<ImageHere>" 标志
for item in data:
    for conversation in item['conversations']:
        if conversation['from'] == 'user':
            conversation['value'] = conversation['value'].replace('<ImageHere>', '')

# 将修改后的数据写回文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"处理完成。修改后的数据已保存到 {output_file}")