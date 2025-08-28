import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import os
import pandas as pd

SYSTEM_PROMPT = (
    """
你是一位经验丰富的农药配方研发专家。
你的任务是根据用户提供的原始要求（PROMPT）和AI模型生成的农药配方（RESPONSE），对RESPONSE进行专业、客观的打分。
请你站在评审的角度，严格遵循以下评分标准对每个方面进行0-2分的打分，并给出最终的总分（0-10分），同时附上详细的理由。
评分标准：
- **原药含量合理性 (0-2分)**：确保百分比之和为1，原药含量准确。
- **辅料选择合理性 (0-2分)**：选用合适、丰富的辅料，合理搭配不同功能的辅料（如分散剂与润湿剂）。
- **配方实用性 (0-2分)**：配方可行性、成本效益、符合工业生产标准。
- **配方详细程度 (0-2分)**：明确标注成分功能，详细准确描述配方特点。
- **质量控制和应用指导 (0-2分)**：包含质量控制指标和具体应用指导。
请务必按照以下格式输出你的评估结果：
总分: [总分数，0-10之间的整数]
只输出上述格式的内容，不包含任何额外的前言或后语。
"""
).strip()

USER_CONTENT = (
    """
以下是用户的生成要求以及可用助剂：
<PROMPT>
{prompt}
</PROMPT>
以下是AI模型所输出的农药配方：
<RESPONSE>
{response_text}
</RESPONSE>
"""
).strip()


def request_hanka_api(system_msg, user_msg):
    import http.client
    import json
    API_KEY = os.environ['API_KEY']
    conn = http.client.HTTPSConnection("blt.to-ai.top")
    payload = json.dumps({
        "model": "deepseek-v3-1-250821-thinking",
        "messages": [
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": user_msg
            }
        ],
        "temperature": 0,
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    conn.request(
        "POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    data = res.read()
    text = data.decode("utf-8")
    return text


def request_dmx_api(system_msg, user_msg):
    import json
    import requests
    API_KEY = os.environ['API_KEY']
    from openai import OpenAI

    # 初始化客户端
    client = OpenAI(
        api_key=API_KEY,  # 替换为你的API Key
        base_url="https://www.dmxapi.cn/v1"  # DMXAPI中转地址
    )

    # 创建对话
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": user_msg
            }
        ],
        model="glm-4.5-free",  # 指定模型
    )

    text = chat_completion

    return text
