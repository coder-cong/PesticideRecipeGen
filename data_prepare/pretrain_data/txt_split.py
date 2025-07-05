# 将txt中的文本分割成1000左右长度的小段，每段向后找第一个句号避免截断

import json

def split_text(input_file, output_file, segment_length=1000):
    """
    将中文文本文件分割成指定长度的段落，并在段尾寻找句号避免截断，
    最后将结果以 JSON 格式输出到文件。

    Args:
        input_file (str): 输入的 TXT 文件路径。
        output_file (str): 输出的 JSON 文件路径。
        segment_length (int): 每个段落的大致长度。
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"错误：文件 {input_file} 未找到。")
        return

    segments = []
    start_index = 0
    while start_index < len(text):
        end_index = min(start_index + segment_length, len(text))

        # 在当前段落末尾附近寻找第一个句号
        search_end = min(end_index + 1000, len(text))  # 向后多搜索一些字符
        dot_index = -1
        for i in range(end_index, search_end):
            if text[i] == '。' or text[i] == '？' or text[i] == '！':
                dot_index = i + 1
                break

        if dot_index != -1:
            segments.append({"text": text[start_index:dot_index],"len":dot_index - start_index})
            start_index = dot_index
        else:
            # 如果附近没有句号，则直接按指定长度截断
            segments.append({"text": text[start_index:end_index],"len":end_index - start_index})
            start_index = end_index

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=4)
        print(f"文本已成功分割并保存到 {output_file}")
    except IOError:
        print(f"错误：无法写入文件 {output_file}。")

if __name__ == "__main__":
    book1 = "C:\\Users\\Cong\\Desktop\\data\\pesticide\\OCR\\[OCR]_农药制剂学 (王开运主编) (Z-Library)_20250507_0945.p.txt"  # 请替换为你的输入 TXT 文件名
    book1_output_file = "农药制剂学_split.json"  # 输出的 JSON 文件名
    split_text(book1, book1_output_file)
    patent = "C:\\Users\\Cong\\Desktop\\data\\pesticide\\patent.txt"
    patent_output_file = "专利_split.json"
    split_text(patent,patent_output_file)