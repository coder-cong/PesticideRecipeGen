"""
从gemini 2.5 pro中蒸馏数据
"""
import os
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
# from util.utils import connect_to_mysql
# from util.utils import extract_and_parse_json


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


def distill_multi_worker(prompts_file, formulation_names_file, adjuvants_file):
    """
    每种剂型使用一个线程来蒸馏
    """
    with open(prompts_file, "r", encoding="utf-8") as f:
        PROMPTS = json.load(f)
    with open(formulation_names_file, "r", encoding="utf-8") as f, open(adjuvants_file, "r", encoding="utf-8") as adjs:
        types = json.load(f)
        type_key_list = list(types)
        adjuvants = json.load(adjs)
    with ThreadPoolExecutor(max_workers=len(type_key_list)) as executor:
        print("主线程开始")
        futures = []
        for i in range(len(type_key_list)):
            futures.append(executor.submit(distill_formulation,
                           i, types, type_key_list, adjuvants, PROMPTS))
        # 使用 as_completed 来获取结果
        # 哪个任务先完成，这个循环就先处理哪个 future
        for future in as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f"一个任务生成了异常: {exc}")
    print("所有线程均已完成")


def distill_formulation(type_idx, types, type_key_list, adjuvants, PROMPTS):
    """
    增加从断点处继续蒸馏：使用process文件记录蒸馏进度，包括剂型和配方名
    每次开始读取上次蒸馏的位置然后继续蒸馏
    """
    pesticide_type = type_key_list[type_idx]
    formulations = types[pesticide_type]
    save_path = f"/root/projs/PesticideRecipeGen/data_prepare/distill_data/out/distill_formulation_{pesticide_type}.json"
    log_path = f"/root/projs/PesticideRecipeGen/data_prepare/distill_data/process/{pesticide_type}"
    if os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            process = json.load(f)
    else:
        process = {
            "formulation": 0
        }
    formulation_idx = process["formulation"]
    print(
        f"Resume from {pesticide_type},{formulation_idx}:{formulations[formulation_idx]}")
    try:
        result = []
        adj_list = concat_adjs(adjuvants[pesticide_type])
        while formulation_idx < len(types[pesticide_type]):
            formulation = types[pesticide_type][formulation_idx]
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
                {"type": pesticide_type, "formulation_idx": formulation_idx, "formulation": formulation, "prompt": prompt, "response": chat_completion.choices[0].message.content})
            formulation_idx += 1
        save_distill_formulation_result(
            save_path, result)
    except Exception as e:
        print("出错了，蒸馏进度保存到了process文件中！！！！")
        print(e)
    process["formulation"] = formulation_idx
    with open(log_path, "w", encoding="utf-8") as log:
        json.dump(process, log, ensure_ascii=False, indent=4)
    save_distill_formulation_result(
        save_path, result)


def save_distill_formulation_result(file_path, result):
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            old_data = json.load(f)
    else:
        old_data = []
    old_data.extend(result)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(old_data, f, ensure_ascii=False, indent=4)


def distill_data_to_Alpaca(data_dir, save_dir):
    files = os.listdir(data_dir)
    result = []
    for data_file in files:
        path = os.path.join(data_dir, data_file)
        with open(path, "r", encoding="utf-8") as f:
            distill_data = json.load(f)
            for formulation in distill_data:
                data = {}
                data["instruction"] = formulation["prompt"]
                data["output"] = formulation["response"]
                result.append(data)
    with open(os.path.join(save_dir, "distill_data_alpaca.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    firms = get_firms_str(
        "/root/projs/PesticideRecipeGen/data_prepare/distill_data/adjuvant_producer.txt")
    # distill_adjuvants_from_firm(
    #     "C:/Projs/PesticideRecipeGen/data/distill/type_adjuvant.json", firms)
    # distill_formulation(
    #     prompts_file="/root/projs/PesticideRecipeGen/data_prepare/distill_data/prompts.json",
    #     formulation_names_file="/root/projs/PesticideRecipeGen/data/distill/formulation_names.json",
    #     adjuvants_file="/root/projs/PesticideRecipeGen/data/distill/Pesticide_adjuvant.json")
    # 多线程蒸馏数据
    # distill_multi_worker(
    #     prompts_file="/root/projs/PesticideRecipeGen/data_prepare/distill_data/prompts.json",
    #     formulation_names_file="/root/projs/PesticideRecipeGen/data/distill/formulation_names.json",
    #     adjuvants_file="/root/projs/PesticideRecipeGen/data/distill/Pesticide_adjuvant.json")
    distill_data_to_Alpaca(
        "/root/projs/PesticideRecipeGen/data_prepare/distill_data/out", "/root/projs/PesticideRecipeGen/data/distill")
