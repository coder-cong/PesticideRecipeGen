import json

def adjust_ids(input_file, output_file, offset):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            if 'id' in data:
                data['id'] += offset
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    # 输入文件名
    input_file = '/home/iiap/下载/tdt_rule.json'
    # 输出文件名
    output_file = 'output.jsonl'
    # 偏移量
    offset = 1000

    adjust_ids(input_file, output_file, offset)
    print(f"All IDs have been adjusted by {offset} and saved to {output_file}.")