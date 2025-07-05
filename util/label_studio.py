import json
import os

def remove_specific_string(input_str):
    # 要移除的特定字符串
    to_remove = "TDT绫波丽复制型，了解。"
    # 使用replace方法去除所有匹配到的字符串
    result_str = input_str.replace(to_remove, "")
    return result_str


def decode_url_encoded_string(text):
    replacements = {
        '%20': ' ',
        '%28': '(',
        '%29': ')',
    }

    for encoded, decoded in replacements.items():
        text = text.replace(encoded, decoded)

    return text


# 读取VQA格式的数据
with open('/home/iiap/PycharmProjects/再次开始的deeplearning/util/json_raw.json', 'r') as f:
    vqa_data = json.load(f)

# 指定图片路径的前缀
prefix_path = "/home/iiap/datasets/internlm_data/VQA/"

# 定义四个固定的问题
questions = [
    "这张图片里的是什么？",
    "接下来可能发生什么情况？",
    "图中的机器人正在做什么？",
    "图中黄色框框选的目标是什么？",
    "图中黄色框框选的区域是什么？",
    "图中黄色框框选的机器人正在做什么？",
    "图中黄色框框选的无人机是否起飞？",
]

# 转换为大模型微调的json格式
converted_data = []
counter = 0


for item in vqa_data:
    # 提取图片路径中的文件名部分
    image_path = decode_url_encoded_string(item["image"])
    image_name = os.path.basename(image_path)

    # 构建完整的图片路径
    full_image_path = os.path.join(prefix_path, image_name)

    for i in range(14):

        converted_item={}

        if i <7:

            if f"answer{i + 1}" in item:
                question = questions[i]
                answer = item[f"answer{i + 1}"]

                answer = remove_specific_string(answer)

                converted_item = {
                    "id": f"vqa_{item['id']}_{i + 1}",
                    "image": full_image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<ImageHere>\n" + question
                        },
                        {
                            "from": "gpt",
                            "value": answer
                        }
                    ]
                }
                counter+=1
                print(converted_item)
        else:

            if f"answer{i + 1}" in item and f"answer{i + 2}" in item and (i+1)%2==0:
                print("自定义消息")
                print(item)
                question = item[f"answer{i + 1}"]
                answer = item[f"answer{i + 2}"]

                answer = remove_specific_string(answer)

                converted_item = {
                    "id": f"vqa_{item['id']}_{i + 1}",
                    "image": full_image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<ImageHere>\n" + question
                        },
                        {
                            "from": "gpt",
                            "value": answer
                        }
                    ]
                }
                counter += 1

            i = i + 1

        if converted_item != {}:

            converted_data.append(converted_item)



print(len(converted_data))
# for i in range(len(converted_data)):
#     print(converted_data[i])
print(counter)

# 将转换后的数据写入json文件
with open('converted_data.json', 'w') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)