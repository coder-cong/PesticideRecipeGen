import re
from bs4 import BeautifulSoup
import json
import requests

base_url = "http://jassbio.com"
# 提取所有的链接


def get_urls():
    with open("./html/WDG_formulation_list.html", "r", encoding="utf-8") as f:
        content = f.read()
        soup = BeautifulSoup(content, "lxml")
        url_list = soup.find(class_="lelm").find_all("a")
    result = []
    for url in url_list:
        result.append(url['href'])
    return result


def request_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(base_url+url, headers=headers, timeout=10)
        response.raise_for_status()
        # requests库会自动猜测网页的编码方式。
        # 如果猜测不准确，可以手动指定：
        response.encoding = 'gbk'  # 或者 'gbk', 'latin-1' 等

        # 返回响应的文本内容，即HTML
        return response.text
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh} - Status Code: {response.status_code}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"An unexpected error occurred: {err}")

    return None


def extract_formulation_list(urls):
    result = {}
    for url in urls:
        name = url.split("/")[2]
        html = request_html(url)
        if html is None:
            print(f"url:{url} is Empty")
            continue
        soup = BeautifulSoup(html, "lxml")
        try:
            text = soup.find(class_="act_neirong").get_text()
        except:
            print(f"url:{url} cannot resolve")
            continue
        text = text.replace("\xa0", "")
        # 定义正则表达式模式
        # ^\s*\d+\.\s*: 匹配行首、可选空白符、数字、点和空白符（序号部分）
        # ([^（]+?): 捕获第一个农药名称（英文名），非贪婪匹配直到遇到开括号'（'
        # （: 匹配开括号
        # ([^）]+): 捕获第二个农药名称（中文名），贪婪匹配直到遇到闭括号'）'
        # ）: 匹配闭括号
        # re.M: 使^和$匹配每一行的开头和结尾，而不是整个字符串的开头和结尾
        pattern = re.compile(r'^\s*\d+[\.、]\s*(.+)', re.M)
        # 查找所有匹配项
        matches = pattern.findall(text)
        # 存储提取到的农药名称
        pesticide_names = []
        # 遍历所有匹配项并格式化输出
        for formulation in matches:
            # 去除英文名称可能存在的首尾空白符
            formulation = formulation.strip()
            pesticide_names.append(formulation)
        result[name] = pesticide_names
    return result


def process_pesticide_str(pesticide_str):
    if pesticide_str is None:
        return [pesticide_str]
    # 首先写出正则表达式
    pattern = re.compile(
        r'^(.*?)\s*([\d\.]+\s*%(?:\s*、\s*[\d\.]+\s*%)*)\s*(.*)$')
    # 使用正则表达式进行匹配
    match = pattern.match(pesticide_str)
    if not match:
        return [pesticide_str]
    prefix = match.group(1).strip()
    full_percentages_str = match.group(2)
    suffix = match.group(3).strip()
    all_percentages = full_percentages_str.split("、")
    result = []
    for percentage in all_percentages:
        result.append(prefix.strip()+" "+percentage.strip()+" "+suffix.strip())
    return result


def process_multi_precentages(file):
    with open(file, "r", encoding="utf-8") as f:
        pesticide_names = json.load(f)
        for pesticide_type in pesticide_names:
            pesticide_list = pesticide_names[pesticide_type]
            new_list = []
            for pesticide_name in pesticide_list:
                new_list.extend(process_pesticide_str(pesticide_name))
            pesticide_names[pesticide_type] = new_list
    with open(file.split(".")[0]+"_processed"+".json", "w", encoding="utf-8") as nf:
        json.dump(pesticide_names, nf, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    urls = get_urls()
    # result = extract_formulation_list(urls)
    # for k, v in result.items():
    #     print(f"{k}:{len(v)}")
    # with open("formulations.json", "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)
    process_multi_precentages(
        "C:/Projs/PesticideRecipeGen/data_prepare/distill_data/formulation_names.json")
