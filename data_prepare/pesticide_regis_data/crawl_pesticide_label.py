import requests
from bs4 import BeautifulSoup
import csv
from time import sleep
import random
import brotli
import re


# 爬取每个农药label界面的信息
def get_pesticide_row(page_start=1, page_end=2447):
    # 基础配置
    base_url = "https://www.icama.cn/BasicdataSystem/pesticideRegistration/queryselect.do"
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,ja;q=0.5",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "Content-Type": "application/x-www-form-urlencoded",
        "Cookie": "JSESSIONID=0DD8BC77E0C3E68D847D654D4713F4F6; pageSize=20; pageNo=1",
        "Host": "www.icama.cn",
        "Origin": "https://www.icama.cn",
        "Referer": "https://www.icama.cn/BasicdataSystem/pesticideRegistration/queryselect.do",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0",
        "Sec-Ch-Ua": '"Not(A:Brand";v="99", "Microsoft Edge";v="133", "Chromium";v="133"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
    }
    session = requests.Session()
    session.headers.update(headers)
    # 分页爬取
    for page in range(page_start, page_end + 1):  # 2447页
        print(f"page{page} Processing...")
        data = {
            "pageNo": str(page),
            "pageSize": "20",
            "accOrfuzzy": "2",
            "djzh": "",
            "nymc": "",
            "cjmc": "",
            "sf": "",
            "nylb": "",
            "zhl": "",
            "jx": "",
            "zwmc": "",
            "fzdx": "",
            "syff": "",
            "dx": "",
            "yxcf": "",
            "yxcf_en": "",
            "yxcfhl": "",
            "yxcf2": "",
            "yxcf2_en": "",
            "yxcf2hl": "",
            "yxcf3": "",
            "yxcf3_en": "",
            "yxcf3hl": "",
            "yxqs_start": "",
            "yxqs_end": "",
            "yxjz_start": "",
            "yxjz_end": "",
        }
        try:
            # 带重试机制的请求
            for _ in range(3):  # 重试3次
                response = session.post(
                    base_url,
                    # headers=headers,
                    data=data,
                    timeout=30,
                )
                if response.status_code == 200:
                    break
                else:
                    print(f"Page {page} Request Failed")
                    sleep(5)  # 延迟5秒再重试
                    continue

                # 新增br解码处理
            if response.status_code == 200:
                if response.headers.get("Content-Encoding") == "br":
                    try:
                        # 自动解码失败时手动处理
                        response._content = brotli.decompress(response.content)
                        response.encoding = "utf-8"  # 强制设置编码
                    except:
                        # 如果服务器声明br但实际未压缩
                        pass

                # 解析HTML
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"id": "tab"})

            if not table:
                print(f"Page {page} Table Not Found")
                continue

            # 提取数据行
            for row in table.find_all("tr")[1:]:  # 跳过标题行
                cols = row.find_all("td")
                if len(cols) != 7:
                    continue

                # 提取一行数据
                djzh = cols[0].get_text(strip=True)
                nymc = cols[1].get_text(strip=True)
                nylb = cols[2].get_text(strip=True)
                jx = cols[3].get_text(strip=True)
                zhl = cols[4].get_text(strip=True)
                yxqz = cols[5].get_text(strip=True)
                cjmc = cols[6].get_text(strip=True)

                data = {
                    "pdno": djzh,
                }

                label_url = "https://www.icama.cn/BasicdataSystem/pesticideRegistration/tagview.do"
                # 带重试机制的请求
                for _ in range(3):  # 重试3次
                    response = session.post(
                        label_url,
                        data=data,
                        timeout=30,
                    )
                    if response.status_code == 200:
                        break
                    else:
                        print(f"Page {page} Request Failed")
                        sleep(5)  # 延迟5秒再重试
                        continue

                # 新增br解码处理
                if response.status_code == 200:
                    if response.headers.get("Content-Encoding") == "br":
                        try:
                            # 自动解码失败时手动处理
                            response._content = brotli.decompress(response.content)
                            response.encoding = "utf-8"  # 强制设置编码
                        except:
                            # 如果服务器声明br但实际未压缩
                            pass

                label_data = parse_label_detail(response.text)

                yield (djzh, label_data)

            # 随机延迟防止封禁
            # sleep(random.uniform(1, 3))

        except Exception as e:
            print(f"Page {page} Error: {str(e)}")
            continue

def html_table_to_markdown(table):
    """将HTML表格转换为Markdown格式"""
    output = []
    headers = []
    rows = []

    # 提取表头和数据行
    for idx, tr in enumerate(table.find_all('tr')):
        cells = [td.get_text(strip=True) for td in tr.find_all(['th', 'td'])]
        if idx == 0:  # 假设第一行为表头
            headers = cells
        else:
            rows.append(cells)

    # 生成Markdown表格
    if headers:
        # 表头行
        output.append("| " + " | ".join(headers) + " |")
        # 分隔线
        output.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # 数据行
    for row in rows:
        output.append("| " + " | ".join(row) + " |")

    return '\n'.join(output)

def get_label_by_num(nums:list):
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,ja;q=0.5",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "Content-Type": "application/x-www-form-urlencoded",
        "Cookie": "JSESSIONID=0DD8BC77E0C3E68D847D654D4713F4F6; pageSize=20; pageNo=1",
        "Host": "www.icama.cn",
        "Origin": "https://www.icama.cn",
        "Referer": "https://www.icama.cn/BasicdataSystem/pesticideRegistration/queryselect.do",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0",
        "Sec-Ch-Ua": '"Not(A:Brand";v="99", "Microsoft Edge";v="133", "Chromium";v="133"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
    }
    session = requests.Session()
    session.headers.update(headers)

    for djzh in nums:
        data = {
            "pdno": djzh,
        }

        label_url = "https://www.icama.cn/BasicdataSystem/pesticideRegistration/tagview.do"
        # 带重试机制的请求
        for _ in range(3):  # 重试3次
            response = session.post(
                label_url,
                data=data,
                timeout=30,
            )
            if response.status_code == 200:
                break
            else:
                sleep(5)  # 延迟5秒再重试
                continue

        # 新增br解码处理
        if response.status_code == 200:
            if response.headers.get("Content-Encoding") == "br":
                try:
                    # 自动解码失败时手动处理
                    response._content = brotli.decompress(response.content)
                    response.encoding = "utf-8"  # 强制设置编码
                except:
                    # 如果服务器声明br但实际未压缩
                    pass

        label_data = parse_label_detail(response.text)

        yield (djzh, label_data)


def parse_label_detail(html_content):
    """解析农药标签详情页面（改进版）"""
    soup = BeautifulSoup(html_content, 'html.parser')
    result = {
        "scope_of_use_and_usage": "",
        "product_performance": "",
        "things_to_note": "",
        "first_aid_measures": "",
        "storage_and_transportation_methods": ""
    }
    # 定位主容器
    main_table = soup.find('table', class_='kuang')
    if not main_table:
        return result

        # 提取使用范围和使用方法（转换为Markdown表格）
    usage_header = main_table.find(lambda tag:
                                       tag.name == 'div' and '使用范围和使用方法：' in tag.text)
    if usage_header:
        usage_table = usage_header.find_next('table')
        if usage_table:
                # 转换为Markdown表格
            result["scope_of_use_and_usage"] = html_table_to_markdown(usage_table)

    # 提取产品性能（带格式清理）
    performance_header = main_table.find(lambda tag:
                                         tag.name == 'span' and '产品性能:' in tag.text)
    if performance_header:
        content = performance_header.find_next('span').get_text(' ', strip=True)
        result["product_performance"] = content.replace('\n', ' ').strip()

    # 提取注意事项（带格式清理）
    note_header = main_table.find(lambda tag:
                                  tag.name == 'span' and '注意事项：' in tag.text)
    if note_header:
        content = note_header.find_next('span').get_text(' ', strip=True)
        result["things_to_note"] = content.replace('\n', ' ').strip()

    # 提取中毒急救措施（带格式清理）
    first_aid_header = main_table.find(lambda tag:
                                       tag.name == 'span' and '中毒急救措施：' in tag.text)
    if first_aid_header:
        content = first_aid_header.find_next('span').get_text(' ', strip=True)
        result["first_aid_measures"] = content.replace('\n', ' ').strip()

    # 提取储存和运输方法（带格式清理）
    storage_header = main_table.find(lambda tag:
                                     tag.name == 'span' and '储存和运输方法：' in tag.text)
    if storage_header:
        content = storage_header.find_next('span').get_text(' ', strip=True)
        result["storage_and_transportation_methods"] = content.replace('\n', ' ').strip()

    return result


def write_to_csv(cols: list, row: list):
    # CSV文件设置
    with open("pesticide_data.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "登记证号",
                "农药名称",
                "农药类别",
                "剂型",
                "总含量",
                "有效期至",
                "登记证持有人",
            ]
        )


if __name__ == "__main__":
    import pymysql

    # 在基础配置后新增数据库配置
    db_config = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': '12345',
        'database': 'pesticide_data',
        'charset': 'utf8mb4'
    }
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()
    # 修改后的SQL语句（添加防重复插入）
    insert_label_sql = """INSERT INTO pesticide_label 
    (pesticide_regis_num, scope_of_use_and_usage, product_performance, 
     things_to_note, first_aid_measures, storage_and_transportation_methods)
    SELECT %s, %s, %s, %s, %s, %s 
    FROM DUAL 
    WHERE NOT EXISTS (
        SELECT 1 
        FROM pesticide_label 
        WHERE pesticide_regis_num = %s
    )"""
    # 数据库插入部分调整：
    # 1134 1305数据需要重新跑一下
    #(1421,2444)
    # log输出目录：C:\Users\gl174\AppData\Local\Programs\PyCharm Professional\bin\label.log
    #0 1800 1950
    # for idx, row in enumerate(get_pesticide_row(page_start=2447, page_end=2447)):
    for idx, row in enumerate(get_label_by_num(['PD20180693','PD20180694',
'PD20180695', 'PD20180696', 'PD20180697', 'PD20180698', 'PD20180699','PD20180700', 'PD20180701',
'PD20180702', 'PD20180703','PD20180704', 'PD20180705', 'PD20180706', 'PD20180707',
'PD20180708', 'PD20180709', 'PD20180710', 'PD20180711',
])):

        try:
            djzh, label_data = row  # 解包返回的元组
            cursor.execute(insert_label_sql, (
                djzh,
                label_data["scope_of_use_and_usage"],
                label_data["product_performance"],
                label_data["things_to_note"],
                label_data["first_aid_measures"],
                label_data["storage_and_transportation_methods"],
                djzh  # 防重复检查的最后一个参数
            ))

            # 每20次操作提交一次

            conn.commit()
        except Exception as e:
            print(f"Error inserting row {idx}: {e}")

    conn.commit()
    cursor.close()
    conn.close()