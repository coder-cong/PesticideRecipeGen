'''给分割之后的文本进行打分，从下面几个方面进行打分
流畅性 (Fluency): 文本是否自然、易读。
连贯性 (Coherence): 文本的逻辑是否清晰，句子之间是否衔接自然。
信息量 (Informativeness): 文本是否包含有用的信息。
可理解性 (Understandability)：文本是否结构清晰，语言简洁明了，逻辑性强，没有混乱难以理解的部分'''


import os
from openai import OpenAI
import json

client = OpenAI(
    api_key = os.environ.get("ARK_API_KEY"),
    base_url = "https://ark.cn-beijing.volces.com/api/v3",
)

# Non-streaming:
completion = client.chat.completions.create(
    model = "deepseek-r1-250120",  # your model endpoint ID
    messages = [
        {"role": "system", "content": '''下面我会给你一端来自农药专业教材农药制剂学的文本，请基于以下几点对文本进行打分，要求直接给出文本的得分，不需要其他内容，满分8分，每点0~2分：
        1.流畅性 (Fluency): 文本是否自然、易读。
        2.连贯性 (Coherence): 文本的逻辑是否清晰，句子之间是否衔接自然。
        3.信息量 (Informativeness): 文本是否包含有用的信息。
        4.可理解性 (Understandability)：文本是否结构清晰，语言简洁明了，逻辑性强，没有混乱难以理解的部分"},
        {"role": "user", "content": "常见的十字花科植物有哪些？'''},
    ],
)
print(completion.choices[0].message.content)

def score(file):
    with open(file,"r",encoding="utf-8") as f:
        obj = json.load(f)
        for text in obj:
            # Non-streaming:
            completion = client.chat.completions.create(
            model = "deepseek-r1-250120",  # your model endpoint ID
            messages = [
        {"role": "system", "content": '''下面我会给你一端来自农药专业教材农药制剂学的文本，请基于以下几点对文本进行打分，要求直接给出文本的得分，不需要其他内容，满分8分，每点0~2分：
        1.流畅性 (Fluency): 文本是否自然、易读。
        2.连贯性 (Coherence): 文本的逻辑是否清晰，句子之间是否衔接自然。
        3.信息量 (Informativeness): 文本是否包含有用的信息。
        4.可理解性 (Understandability)：文本是否结构清晰，语言简洁明了，逻辑性强，没有混乱难以理解的部分"},
        {"role": "user", "content": "常见的十字花科植物有哪些？'''},
    ],
)
            if float(completion.choices[0].message.content)<5:
                print(completion.choices[0].message.content)
                return
    
if __name__ == "__main__":
    file = "C:\\Users\\Cong\\Desktop\\myProj\\PesticideRecipeGen\\data_prepare\\pretrain_data\\农药制剂学_split.json"
    score(file)
