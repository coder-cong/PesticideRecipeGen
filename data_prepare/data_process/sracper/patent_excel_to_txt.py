import pandas as pd

# 读取Excel文件
file_path = r'E:\drug\sracper\google_patent.xlsx'
df = pd.read_excel(file_path)

# 提取C列和F列的数据
column_c = df.iloc[:, 2]  # C列（索引从0开始，所以C列是第2列）
column_f = df.iloc[:, 5]  # F列（索引从0开始，所以F列是第5列）

# 将数据组合成空格隔开的字符串
result = [f"{c} {f}" for c, f in zip(column_c, column_f)]

# 输出结果到txt文件
output_file_path = r'E:\drug\sracper\output.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    for line in result:
        file.write(line + '\n')

print(f"数据已写入到 {output_file_path}")
