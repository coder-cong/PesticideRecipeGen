'''给分割之后的文本进行打分，从下面几个方面进行打分
流畅性 (Fluency): 文本是否自然、易读。
连贯性 (Coherence): 文本的逻辑是否清晰，句子之间是否衔接自然。
信息量 (Informativeness): 文本是否包含有用的信息。
可理解性 (Understandability)：文本是否结构清晰，语言简洁明了，逻辑性强，没有混乱难以理解的部分'''


import os
from openai import OpenAI
import json
import re
import threading
import re
from concurrent.futures import ThreadPoolExecutor

print(os.environ.get("ARK_API_KEY"))
client = OpenAI(
    api_key = os.environ.get("ARK_API_KEY"),
    base_url = "https://ark.cn-beijing.volces.com/api/v3",
)

# Non-streaming:
# completion = client.chat.completions.create(
#     model = "deepseek-r1-250120",  # your model endpoint ID
#     messages = [
#         {"role": "system", "content": '''下面我会给你一端来自农药专业教材农药制剂学的文本，请基于以下几点对文本进行打分，要求直接给出各个点的得分，最后给出总体得分，不需要其他内容，以json字符串格式返回得分，例如{"Fluency":1,"Coherence":1,"Informativeness":2,"Understandability":2,"total":6}，满分8分，每点0~2分：
#         1.流畅性 (Fluency): 文本是否自然、易读。
#         2.连贯性 (Coherence): 文本的逻辑是否清晰，句子之间是否衔接自然。
#         3.信息量 (Informativeness): 文本是否包含有用的信息。
#         4.可理解性 (Understandability)：文本是否结构清晰，语言简洁明了，逻辑性强，没有混乱难以理解的部分'''},
#         {"role": "user", "content": "常见的十字花科植物有哪些？"},
#     ],
# )
# print(completion.choices[0].message.content)
# 使用多线程进行打分
def score(file,num_workers=10):
    results = []
    with open(file,"r",encoding="utf-8") as f:
        data = json.load(f)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(score_text, idx, text) for idx, text in enumerate(data)]
            for future in futures:
                result = future.result()
                if result is not None:
                    results.append(result)
    return results

# 给一段文本打分
def score_text(idx,data):
    # Non-streaming:
            completion = client.chat.completions.create(
            model = "deepseek-r1-250120",  # your model endpoint ID
            messages = [
        {"role": "system", "content": '''下面我会给你一端来自农药专业教材农药制剂学的文本，请基于以下几点对文本进行打分，要求直接给出各个点的得分，最后给出总体得分，不需要其他内容，以json字符串格式返回得分，例如{"Fluency":1,"Coherence":1,"Informativeness":2,"Understandability":2,"total":6}，满分100分，每点25分：
        1.流畅性 (Fluency): 文本是否自然、易读。
        2.连贯性 (Coherence): 文本的逻辑是否清晰，句子之间是否衔接自然。
        3.信息量 (Informativeness): 文本是否包含有用的信息。
        4.可理解性 (Understandability)：文本是否结构清晰，语言简洁明了，逻辑性强，没有混乱难以理解的部分'''},
        {"role": "user", "content": data['text']},
    ],
)
            # 使用正则表达式匹配 JSON 格式的字符串
            match = re.search(r'```json\n(.*)\n```', completion.choices[0].message.content, re.DOTALL)

            if match:
                json_string = match.group(1)
                try:
                    response = json.loads(json_string)
                    print(response)
                    if 'total' in response:
                        print(f"线程：{threading.current_thread().name} {idx}得分{response['total']}")
                        if response['total'] > 70:
                            return {"text":data['text']}
                    else:
                        print(f"线程：{threading.current_thread().name}{idx}json对象格式不正确")
                except json.JSONDecodeError as e:
                    print(f"JSON decoding error: {e}")
                    print(f"Failed to decode string: '{json_string}'")
            else:
                print(f"线程：{threading.current_thread().name}{idx}未找到符合 JSON 格式的字符串。")
            
            return None
    
if __name__ == "__main__":
    file = "./process_out/农药制剂学_split.json"
    filtered_res = score(file)
    print(f"过滤后结果数：{len(filtered_res)}")
    # 保存过滤后结果
    with open("./process_out/农药制剂学_filtered.json",mode="w",encoding="utf-8") as f:
        json.dump(filtered_res,f,indent=4,ensure_ascii=False)