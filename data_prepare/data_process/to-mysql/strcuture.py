def process_patent_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    patents = []
    current_patent = []
    current_patent_title = ""

    # 读取文件内容并按每个专利进行分割
    for line in lines:
        if line.startswith('专利'):
            if current_patent:
                patents.append((current_patent_title, ''.join(current_patent)))
                current_patent = []
            current_patent_title = line.strip()
        else:
            current_patent.append(line)
    if current_patent:
        patents.append((current_patent_title, ''.join(current_patent)))

    # 处理每个专利并分割内容
    processed_patents = []
    for title, patent in patents:
        # 找到第一个“技术领域”
        tech_field_index = patent.find('技术领域')
        
        if tech_field_index != -1:
            # 找到第一个“技术领域”之前的内容
            before_tech_field = patent[:tech_field_index]
            after_tech_field = patent[tech_field_index:]
            
            # 找到最后一个“。”的位置
            last_period_index = before_tech_field.rfind('。')
            
            if last_period_index != -1:
                before_last_period = before_tech_field[:last_period_index + 1]
                after_last_period = before_tech_field[last_period_index + 1:]
                
                # 重新组合
                split_patent = "摘要：" + before_last_period + '\n' + "标题：" + after_last_period + after_tech_field
            else:
                # 如果没有“。”，直接加分隔符
                split_patent = "摘要：" + patent
        else:
            split_patent = "摘要：" + patent

        # 只替换每个关键词的第一次出现并加上标签
        sections = ['技术领域', '背景技术', '发明内容', '具体实施方式']
        for section in sections:
            index = split_patent.find(section)
            if index != -1:
                split_patent = split_patent[:index] + f'\n{section}：' + split_patent[index + len(section):]

        processed_patents.append((title, split_patent))

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for title, patent in processed_patents:
            f.write(title + '\n' + patent + '\n')

# 使用示例
input_file = r'E:\drug\sracper\data\patent3839-15438.txt'  # 输入文件名
output_file = r'E:\drug\sracper\data\structure_patent3839-15438.txt'  # 输出文件名
process_patent_file(input_file, output_file)
