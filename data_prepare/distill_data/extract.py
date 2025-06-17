from bs4 import BeautifulSoup
import os
import requests
import json

def extract_table(html):
    """
    从给定的html字符串中提取助剂并保存为键值对
    """
    soup = BeautifulSoup(html,"lxml")
    tables = soup.find_all("table")
    all_table_list = []
    for table in tables:
        # 提取表头
        headers = []
        # 假设第一个tr是表头
        header_row = table.find('tr')
        for th in header_row.find_all('td'): # 表头也是用td表示的
            headers.append(th.get_text(strip=True))
        # 提取数据行
        data = []
        data.append(headers)
        # 用于处理 rowspan 的变量
        current_jixing = None
        # 从第二个tr开始遍历，因为第一个是表头
        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            row_values = []
            # 检查第一个单元格是否有 rowspan 属性
            if 'rowspan' in cells[0].attrs:
                # 如果有 rowspan，说明这是一个新的“剂型”值
                current_jixing = cells[0].get_text(strip=True)
                # 当前行的第一个值就是这个新的“剂型”，然后处理从第二个单元格开始的剩余数据
                row_values.append(current_jixing)
                for cell in cells[1:]:
                    row_values.append(cell.get_text(strip=True))
            else:
                # 如果没有 rowspan，说明它属于上一个“剂型”组
                # 所以当前行的第一个值就是之前保存的 current_jixing
                row_values.append(current_jixing)
                for cell in cells: # 处理所有单元格
                    row_values.append(cell.get_text(strip=True))
            
            # 将提取到的行数据添加到总数据列表中
            data.append(row_values)
        all_table_list.append(data)
    return all_table_list
def get_html_content(url):
    """
    请求指定URL的HTML内容。
    Args:
        url (str): 目标网页的URL。
    Returns:
        str: 网页的HTML内容，如果请求失败则返回None。
    """
    # 模拟浏览器User-Agent，防止某些网站拒绝Python爬虫
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        # 发送GET请求，设置超时时间为10秒
        # verify=True 默认会验证SSL证书，如果是自签名证书或本地测试环境可能需要设置为False
        response = requests.get(url, headers=headers, timeout=10)
        # 检查HTTP状态码。如果状态码是200 (OK) 则继续，否则会抛出HTTPError异常
        response.raise_for_status()
        # requests库会自动猜测网页的编码方式。
        # 如果猜测不准确，可以手动指定：
        response.encoding = 'gbk' # 或者 'gbk', 'latin-1' 等
        
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

def extract_to_json():
    """
    请求到html并处理成表格每行单独一个列表的形式，未处理成键值对
    """
    base_url = "http://jassbio.com/"
    # 记录所有助剂的链接
    all_types_url = []
    # 获取到所有助剂页面的链接
    with open("./WDG.html","r",encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(),'lxml')
        all_div = soup.find_all("div",class_="lelm")
        all_a = all_div[0].find_all("a")
        for a in all_a:
            all_types_url.append(a['href'])
    print(all_types_url)
    all_aux = []
    for url in all_types_url:
        html = get_html_content(base_url+url)
        if html is not None:
            tables = extract_table(html)
            all_aux.extend(tables)
    with open("output.json",mode="w",encoding="utf-8") as f:
        json.dump(all_aux,f,ensure_ascii=False)

def process_to_kv():
    """
    将extract_to_json获取到的json形式的数据做进一步处理
    """
    with open("output.json","r",encoding="utf-8")

if __name__ == "__main__":
   pass 