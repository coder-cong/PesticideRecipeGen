import json
#对输出的数据再处理，去除空行以及多余字符
def filter_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                # 尝试解析JSON对象
                json_object = json.loads(line)
                # 如果对象不是空的（如 []），则写入输出文件
                if json_object:
                    json.dump(json_object, outfile, ensure_ascii=False)
                    outfile.write('\n')
            except json.JSONDecodeError:
                # 如果解析失败，跳过该行
                continue

# 使用示例
input_file = r'E:\drug\sracper\data\fromPatent1-300byDeepseek.jsonl'
output_file = r'E:\drug\sracper\data\fromPatent1-300byDeepseek-out.jsonl'
filter_jsonl(input_file, output_file)
