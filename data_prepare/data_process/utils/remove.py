
#去除多余的换行符

def remove_extra_newlines(input_file, output_file):
    """
    读取输入文件，去除多余的换行符，并将结果保存到输出文件。
    """
    try:
        # 打开输入文件并读取内容
        with open(input_file, 'r', encoding='utf-8') as infile:
            content = infile.read()

        # 替换多余的换行符
        # 这里使用正则表达式，将连续的换行符替换为单个换行符
        import re
        cleaned_content = re.sub(r'\n\s*\n', '\n', content).strip()

        # 将处理后的内容写入输出文件
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(cleaned_content)

        print(f"处理完成！已去除多余换行符，结果保存到 {output_file}")
    except FileNotFoundError:
        print(f"错误：文件 {input_file} 未找到。")
    except Exception as e:
        print(f"处理时发生错误：{e}")


# 示例用法
input_file = r'E:\drug\sracper\abstract_output.txt'  # 输入文件名
output_file = r'E:\drug\sracper\abstract_output.txt'  # 输出文件名
remove_extra_newlines(input_file, output_file)