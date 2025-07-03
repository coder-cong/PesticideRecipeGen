"""
从gemini 2.5 pro中蒸馏数据
"""
import os
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


def distill_formulation(prompts_file, formulation_names_file, adjuvants_file):
    with open(prompts_file, "r", encoding="utf-8") as f:
        PROMPTS = json.load(f)
    with open(formulation_names_file, "r", encoding="utf-8") as f:
        with open(adjuvants_file, "r", encoding="utf-8") as adjs:
            adjuvants = json.load(adjs)
            types = json.load(f)
            for pesticide_type in types:
                result = []
                adj_list = concat_adjs(adjuvants[pesticide_type])
                for formulation in types[pesticide_type]:
                    prompt = PROMPTS[pesticide_type].format(
                        name=formulation, adjuvants=adj_list)
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        # 替换成你先想用的模型全称， 模型全称可以在DMXAPI 模型价格页面找到并复制。
                        model="gemini-2.5-pro-thinking",
                    )
                    print(chat_completion.choices[0].message.content)
                    result.append(
                        {"prompt": prompt, "response": chat_completion.choices[0].message.content})
                with open(f"distill_formulation_{pesticide_type}.json", "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=4)
                    print(f"{pesticide_type} completed")


if __name__ == "__main__":
    firms = get_firms_str(
        "C:/Projs/PesticideRecipeGen/data_prepare/distill_data/adjuvant_producer.txt")
    # distill_adjuvants_from_firm(
    #     "C:/Projs/PesticideRecipeGen/data/distill/type_adjuvant.json", firms)
    distill_formulation(
        prompts_file="C:/Projs/PesticideRecipeGen/data_prepare/distill_data/prompts.json",
        formulation_names_file="C:/Projs/PesticideRecipeGen/data/distill/formulation_names.json",
        adjuvants_file="C:/Projs/PesticideRecipeGen/data/distill/Pesticide_adjuvant.json")
