from openai import OpenAI
import time
from typing import List
import os

#------------------------------------------------------------#
#      生成数据集，原始格式，保存在pesticide_knowledge_summary.txt中,如有重复，建立一个新的txt文件
#------------------------------------------------------------#

# 初始化客户端
# 注意：解析 https://api.deepseek.com 时遇到了问题，可能是链接本身的问题，也可能是网络连接问题。
# 请检查链接的合法性，并确保网络连接正常。如果问题仍然存在，请联系 API 提供者。
# client = OpenAI(
#     api_key=os.getenv("MOONSHOT_API_KEY", "sk-UuK21E7HPqObfBdCBFvTHnfCxr7pTOsXPFuWNkVmpBKHxYwd"),
#     base_url="https://api.moonshot.cn/v1"
# )

client = OpenAI(
    api_key="sk-42110a936b8949169455a5553089e2f4",
    base_url="https://api.deepseek.com"
) 
# client = OpenAI(
#     api_key="23158383-5719-425a-a378-55e719f33d43",
#     base_url="https://ark.cn-beijing.volces.com/api/v3"
# )

# 读取文本文件

def read_file_in_chunks(file_path: str, chunk_size: int = 2000) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        buffer = ""
        for line in file:
            buffer += line
            while len(buffer) >= chunk_size:
                # 找到第一个句号
                period_index = buffer.find('。', chunk_size)
                if period_index != -1:
                    yield buffer[:period_index + 1]
                    buffer = buffer[period_index + 1:]
                else:
                    yield buffer[:chunk_size]
                    buffer = buffer[chunk_size:]
        if buffer:
            yield buffer



# 调用API进行处理
def process_chunk(chunk: str, prompt: str) -> str:
    try:
        # 调用API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 # 提取信息的prompt
                 # "content": "您是一位农药领域的专家助手，专注于提取和总结农药和农药配方相关的知识。您的任务是从文本中提取相关信息，进行总结，将其整理成适合大模型微调的数据集。这个数据集是json格式，除了数据集外，不需要多余的字母跟符号，一定要注意！！若输入的文字可以总结成多条独立的数据，那么每条数据都是一条json数据。关于格式，数据集的格式类似于Alpaca数据集格式，包含‘instruction（指令）’‘input（输入，可选）’‘output（输出）’三部分,这三部分不能缺少。这三个部分的解释如下，instruction: 任务的指令，告诉模型需要完成什么操作，需要明确任务的具体范围或目标。input: 任务所需的输入。如果任务是开放式的或者不需要明确的输入，这一字段可以为空字符串。若内容过长要精简为与任务直接相关的内容。output: 任务的期望输出，也就是模型在给定指令和输入情况下需要生成的内容。一定要注意！！若输入的文字可以总结成多条独立的数据，不要嵌套到output中，output中不要添加过多的括号，形成语意清晰的文字即可。例如：‘instruction’: ‘生成一个适用于小麦防治白粉病的农药配方’, ‘output’: ‘配方名称：配方A；适用作物：小麦；防治病虫害类型：白粉病；有效成分：吡唑醚菌酯 20%；助剂：乳化剂 5%；使用方法：稀释1000倍，抽穗期喷施’。"

                # "content": "您是一位农药领域的专家助手，下述文字是某个或多个农药的详细信息，请你逐行逐字仔细阅读，详细思考，然后从文本中提取以下4类信息(只提取这4类），1、中英文对照 2、加工剂型：如EC、SC等 3、活性用途：描述作用原理 4、用途：作物或防治对象，要求：生成严格符合JSON标准的JSONL文件，使用双引号包裹字段名和值，只提取上述四类信息，每条问答对独立成行，格式为：{'instruction': '问题', 'input': ,'output': '答案'}，每条回答一定要以中文农药试剂名称开始，若某字段无数据，则要跳过生成该问答对（如无剂型则不生成剂型问题），禁止虚构未提及的信息。 示例： {'instruction': '灭草松对应的英文名称是什么？', 'input': '', 'output': 'bentazone'} {'instruction': '联苯菊酯的加工剂型有哪些？', 'input': '', 'output': 'EC、SC'} {'instruction': '氟铃脲的作用机制是什么？', 'input': '', 'output': '抑制昆虫几丁质合成'} {'instruction': '阿维菌素的用途', 'input': '', 'output': '用于观赏植物、棉花、柑橘、梨果、坚果、蔬菜、马铃薯和其他作'}"
                  #   "content": " 您是一位农药领域的专家助手，请仔细阅读以下文本，总结与农药配方相关的信息然后形成合适的问答对。要求：1. 严格按照JSONL格式输出；2. 使用双引号包裹字段名和值；3. 每条JSON数据独立成行；4. 仅提取文本中明确提及的信息，禁止虚构或补充未提及的内容。5.识别配方核心要素：有效成分+剂型（如'25%噻虫嗪悬浮剂'）6.问题必须包含完整技术特征：作物对象+防治对象+剂型（如'防治水稻二化螟的5%阿维菌素乳油配方组成'）7.禁止使用孤立编号（实施例X、表格Y等）8.问答对需自包含完整信息. 例如： {'instruction': '某某农药的配方组成是什么、某农药的应用是什么等等','input': '原始文本内容, 'output': '基于问题的回答，回答的内容来源于文本内容，需要进行一定的总结与整理，禁止虚构 }   您是一位农药领域的专家助手，请仔细阅读以下文本，总结与农药配方相关的信息然后形成合适的问答对。要求：1. 严格按照JSONL格式输出；2. 使用双引号包裹字段名和值；3. 每条JSON数据独立成行；4. 仅提取文本中明确提及的信息，禁止虚构或补充未提及的内容。5.识别配方核心要素：有效成分+剂型（如'25%噻虫嗪悬浮剂'）6.问题必须包含完整技术特征：作物对象+防治对象+剂型（如'防治水稻二化螟的5%阿维菌素乳油配方组成'）7.禁止使用孤立编号（实施例X、表格Y等）8.问答对需自包含完整信息. 例如： {'instruction': '某某农药的配方组成是什么、某农药的应用是什么等等','input': '原始文本内容, 'output': '基于问题的回答，回答的内容来源于文本内容，需要进行一定的总结与整理，禁止虚构 }  "
                     "content": " 您是一位农药领域的专家助手，请仔细阅读以下文本，总结与农药配方相关的信息然后形成合适的问答对。要求：1. 严格按照JSONL格式输出；2. 使用双引号包裹字段名和值；3. 每条JSON数据独立成行；4. 仅提取文本中明确提及的信息，禁止虚构或补充未提及的内容。5.识别配方核心要素：如有效成分+剂型（如'25%噻虫嗪悬浮剂'）6.问题需要包含技术特征：如作物对象+防治对象+剂型（如'防治水稻二化螟的5%阿维菌素乳油配方组成'）7.禁止使用孤立编号（实施例X、表格Y等、专利X等）8.input标签的内容必须为空   例如： {'instruction': '某某农药的配方组成是什么、某农药的应用是什么等等','input': '', 'output': '基于问题的回答，回答的内容来源于文本内容，需要进行一定的总结与整理，禁止虚构 }  "
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
    output_file = r'E:\drug\sracper\data\abs\8200-9xxx-output.txt'
    if os.path.exists(output_file):
        base, extension = os.path.splitext(output_file)
        i = 1
        while os.path.exists(output_file):
            output_file = f"{base}_{i}{extension}"
            i += 1

    # 处理每个文本块
    for chunk in read_file_in_chunks(file_path):
        print(f"Processing chunk: {chunk[:50]}...{chunk[-10:]}")

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
    input_file = r'E:\drug\sracper\data\abs\8200-9xxx-input.txt'
    main(input_file, "")