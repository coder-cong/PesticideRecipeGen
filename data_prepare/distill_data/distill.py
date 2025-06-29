"""
从gemini 2.5 pro中蒸馏数据
"""

import json
from openai import OpenAI
from util.utils import connect_to_mysql
from util.utils import extract_and_parse_json


# ------------------------------------------------------------------------------------
# 在 Openai官方库 中使用 DMXAPI KEY 的例子
# 需要先 pip install openai
# ------------------------------------------------------------------------------------
client = OpenAI(
    api_key="sk-SKuKSKZvrDKkY6QDCBXBKDdZJ7Mf0rhj29YByc5gDN7MbePI",  # 替换成你的 DMXapi 令牌key
    # 需要改成DMXAPI的中转 https://www.dmxapi.cn/v1 ，这是已经改好的。
    base_url="https://www.dmxapi.cn/v1",
)

# PROMPT = """
# 你是一个农药配方生成专家，需要为我生成尽可能详细的农药配方，请确保助剂类型尽可能完备，如果需要的助剂没有给出请自行选择合适的具体的助剂,需要生成的配方名称为：{name}(注意给出的配方名称中可能含有多个不同的有效成分含量，选择其中一个生成即可),可以使用的助剂如下：{adjuvants}
# """


def concat_adjs(adj_list):
    """
    将助剂列表拼接成字符串
    """
    return str(adj_list)


def get_firms_str(firms_file):
    """
    读取文件并将助剂生产厂商拼接为一个字符串
    Args:
        firms_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    result = ""
    with open(firms_file, "r", encoding="utf-8") as f:
        for line in f:
            result += line.rstrip('\n')
            result += '、'
    return result[:-1]


def distill_adjuvants_from_firm(adjuvant_types, firms):
    """
    蒸馏出农药所使用助剂的公司的助剂数据
    """
    PROMPT = """你是一个善于在互联网上收集整理用于农药配方不同剂型所需要相关助剂信息的专家，现在我需要你找出我所要求的农药配方助剂的相关信息，要求如下：用于{type}的{adjuvant}({description})，要求找出如下生产公司的具体助剂（具体到型号，**只需要这些公司的助剂以及相关信息**，不需要其他公司的）{firms}，你需要对找出的具体助剂进行**检验**确保型号**真实存在**,格式要求如下:只需要给我整理之后的助剂json列表，其他信息不需要,json列表中对象的格式要求为包含生产公司(firm)，具体型号(model)，相关信息（info,可选，如果有的话最好加上）"""
    with open(adjuvant_types, "r", encoding="utf-8") as f:
        adj_types = json.load(f)
        for formulation_type in adj_types:
            adj_type_list = adj_types[formulation_type]
            for adj in adj_type_list:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": PROMPT.format(type=formulation_type, adjuvant=adj["name"], description=adj["description"], firms=firms),
                        }
                    ],
                    # 替换成你先想用的模型全称， 模型全称可以在DMXAPI 模型价格页面找到并复制。
                    model="gemini-2.5-pro-preview-06-05",
                )
                # print(chat_completion)
                print(PROMPT.format(type=formulation_type,
                      adjuvant=adj["name"], description=adj["description"], firms=firms))
                return


def distill_formulation():
    PROMPT = """
    你是一个农药配方生成专家，需要为我生成尽可能详细的农药配方，请确保助剂类型尽可能完备，如果需要的助剂没有给出请自行选择合适的具体的助剂,需要生成的配方名称为：{formulation_name}(注意给出的配方名称中可能含有多个不同的有效成分含量，选择其中一个生成即可),可以使用的助剂如下：{adjuvants},注意需要以json格式返回，要求为"{{"name":配方名称,"active_ingredients":[{{"name":有效成分名(具体到型号),"function":作用,"content":xx%}}]}}"
    """
    PROMPT_TEMPLATE = """
    你是一位顶尖的农药配方生成专家。请根据我提供的配方名称和建议的助剂列表，生成一个专业、详细且完整的农药配方。
    **任务要求：**
    1.  **配方名称：** {formulation_name}
    2.  **核心任务：** 基于上述配方名称，设计一个完整的配方。请务必包含名称中提到的所有有效成分及其含量。
    3.  **助剂选择：**
        - 优先从以下【建议助剂列表】中选择合适的助剂。
        - 如果列表中的助剂不适用或不充分，请你凭借专业知识，自行选择并补充其他必要的助剂（如润湿剂、分散剂、乳化剂、增稠剂、防冻剂、溶剂等），以确保配方稳定有效。
        - **建议助剂列表：** {adjuvants}
    4.  **含量要求：** 请确保所有成分（有效成分、所有助剂、载体/溶剂）的百分比含量总和精确到100%。
    5.  **输出格式：** 必须严格按照以下JSON格式返回，不要包含任何额外的解释或说明文字。
    **JSON格式定义：**
    ```json
    {{
    "name": "完整的配方名称",
    "total_content": "100%",
    "active_ingredients": [
        {{
        "name": "有效成分的具体化学名",
        "content": "xx%"
        }}
    ],
    "adjuvants": [
        {{
        "name": "具体的助剂商品名或化学名",
        "type": "助剂类型 (例如: 乳化剂, 润湿剂, 防冻剂)",
        "content": "xx%"
        }}
    ],
    "carrier_or_solvent": {{
        "name": "载体或溶剂的名称 (例如: 水, 溶剂油)",
        "content": "补足至100%的剩余含量"
    }}
    }}
    ```
    **备注**: 除了助剂具体型号和配方名称外，其余尽量使用中文进行输出
    """
    result = {}
    with open("../../data/distill/formulation_names.json", "r", encoding="utf-8") as f:
        with open("../../data/distill/Pesticide_adjuvant.json", "r", encoding="utf-8") as adjs:
            adjuvants = json.load(adjs)
            types = json.load(f)
            for type in types:
                adj_result = []
                result[type] = adj_result
                adj_list = concat_adjs(adjuvants[type])
                for formulation in types[type]:
                    # print(PROMPT_TEMPLATE.format(
                    #     formulation_name=formulation, adjuvants=adj_list))
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": PROMPT_TEMPLATE.format(
                                    formulation_name=formulation, adjuvants=adj_list)
                            }
                        ],
                        # 替换成你先想用的模型全称， 模型全称可以在DMXAPI 模型价格页面找到并复制。
                        model="gemini-2.5-pro-preview-06-05-ssvip",
                    )
                    formulation_obj = extract_and_parse_json(
                        chat_completion.choices[0].message.content)
                    print(formulation_obj)
                    adj_result.append(formulation_obj)
            with open("distill_formulation.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
            return


if __name__ == "__main__":
    firms = get_firms_str(
        "C:/Projs/PesticideRecipeGen/data_prepare/distill_data/adjuvant_producer.txt")
    # distill_adjuvants_from_firm(
    #     "C:/Projs/PesticideRecipeGen/data/distill/type_adjuvant.json", firms)
    distill_formulation()
