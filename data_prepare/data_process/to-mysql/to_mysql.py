import mysql.connector
import re

#结构化的输入文件的第一行必须为空！！！！！！！！！！！！！！！！！！！！！！！！！！！
def connect_to_database():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='forfuture2025',
        database='patent_db'
    )

def parse_patent_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 使用正则表达式找到所有的专利块
    pattern = re.compile(r'(?<=\n)(专利\d+)\n(.*?)(?=\n专利\d+\n|$)', re.S)
    matches = pattern.findall(content)

    patent_data = []

    for match in matches:
        patent_id, patent_content = match
        lines = patent_content.strip().split('\n')
        data = {
            'title': '',
            'abstract': '',
            'tech_field': '',
            'background': '',
            'invention_content': '',
            'detailed_description': ''
        }

        for line in lines:
            line = line.strip()
            if line.startswith('摘要：'):
                data['abstract'] = line.replace('摘要：', '').strip()
            elif line.startswith('标题：'):
                data['title'] = line.replace('标题：', '').strip()
            elif line.startswith('技术领域：'):
                data['tech_field'] = line.replace('技术领域：', '').strip()
            elif line.startswith('背景技术：'):
                data['background'] = line.replace('背景技术：', '').strip()
            elif line.startswith('发明内容：'):
                data['invention_content'] = line.replace('发明内容：', '').strip()
            elif line.startswith('具体实施方式：'):
                data['detailed_description'] = line.replace('具体实施方式：', '').strip()

        # 如果所有字段都存在
        if all(data.values()):
            patent_data.append(data)
        else:
            # 如果有字段缺失，将所有内容存储在详细描述中
            data['detailed_description'] = patent_content.strip()
            data.update({key: '' for key in data if key != 'detailed_description'})
            patent_data.append(data)

    return patent_data

def insert_patents_to_db(patent_data):
    conn = connect_to_database()
    cursor = conn.cursor()

    for index, data in enumerate(patent_data):
        cursor.execute(
            """
            INSERT INTO patents (
                patent_index, title, abstract, tech_field, background, invention_content, detailed_description
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                f"专利{index+1}", data.get('title', ''), data.get('abstract', ''),
                data.get('tech_field', ''), data.get('background', ''),
                data.get('invention_content', ''), data.get('detailed_description', '')
            )
        )

    conn.commit()
    cursor.close()
    conn.close()

def main():
    input_file = r'E:\drug\sracper\data\structure_patent3839-15438.txt'
    patent_data = parse_patent_file(input_file)
    insert_patents_to_db(patent_data)
    print("数据插入成功")

if __name__ == "__main__":
    main()
