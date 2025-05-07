from openai import OpenAI
import time
from typing import List
import os

#------------------------------------------------------------#
#      进一步筛选专利，让llm根据标题判断是否是与农药相关的专利
#------------------------------------------------------------#

# 初始化客户端
# 注意：解析 https://api.deepseek.com 时遇到了问题，可能是链接本身的问题，也可能是网络连接问题。
# 请检查链接的合法性，并确保网络连接正常。如果问题仍然存在，请联系 API 提供者。
# client = OpenAI(
#     api_key=os.getenv("MOONSHOT_API_KEY", "sk-UuK21E7HPqObfBdCBFvTHnfCxr7pTOsXPFuWNkVmpBKHxYwd"),
#     base_url="https://api.moonshot.cn/v1"
# )

client = OpenAI(
    api_key="sk-42110a936b89491695a5553089e2f4",
    base_url="https://api.deepseek.com"
)

# 读取文本文件
def read_file_in_chunks(file_path: str, lines_per_chunk: int = 10) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            lines = [file.readline() for _ in range(lines_per_chunk)]
            if not any(lines):  # 如果所有行都是空的，结束读取
                break
            yield ''.join(lines)


# 调用API进行处理
def process_chunk(chunk: str, prompt: str) -> str:
    try:
        # 调用API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 # 提取信息的prompt

                     #"content": " 下面文字的每行是某种专利的标题以及相应的链接，请你判断该专利是否与农药配方相关，若相关，请你直接返回url，不能进行任何的修改，每行一个。 "
                     "content": " 你是一位专注于农药配方的研究专家。你的任务是从以下专利摘要中筛选出与传统农药配方相关性高的内容。请逐行判断，若相关，直接返回该行数据的url，不要进行修改。输入的数据格式示例如下：专利1 URL: [url内容] 摘要内容。返回的数据格式示例如下：专利1 URL: [url内容] 。你可以参考以下步骤进行分析：1. **识别传统成分**：查找摘要中是否提到常见的传统农药成分，如阿维菌素、吡虫啉、硫磺等。2. **应用方法和用途检查**：检查摘要中描述的应用方法和用途是否与传统农药一致，如喷雾、土壤处理等。3. **排除新型农药配方**：去除涉及新型农药配方的摘要，如光学、微生物等。 "

        },
                {"role": "user", "content": prompt + chunk}
            ],
            temperature=1,
            stream=False
        )

        # 提取回答内容
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        time.sleep(0.05)
        return None


# 主函数
def main(file_path: str, prompt: str) -> None:
    start_time = time.time()
    results: List[str] = []
    # 检查文件是否存在，如果存在则创建新的文件
    output_file = r'E:\drug\sracper\data\abstract_test-out-backlast.txt'
    if os.path.exists(output_file):
        base, extension = os.path.splitext(output_file)
        i = 1
        while os.path.exists(output_file):
            output_file = f"{base}_{i}{extension}"
            i += 1

    # 处理每个文本块
    for chunk in read_file_in_chunks(file_path):
        print(f"Processing chunk: {chunk[:500]}...")  # 打印前50个字符

        # 添加延迟以避免超过API速率限制
        time.sleep(0.05)

        result = process_chunk(chunk, prompt)
        if result:
            results.append(result)


            # 写入结果
            with open(output_file, 'a', encoding='utf-8') as outfile:
                outfile.write(result + "\n")

    print("Processing completed!")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"整个脚本的运行时间为: {elapsed_time:.6f} 秒")


if __name__ == '__main__':
    input_file = r'E:\drug\sracper\urlsAndAbstract-out.txt'
    main(input_file, "")