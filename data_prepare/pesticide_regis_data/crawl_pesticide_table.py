import requests
from bs4 import BeautifulSoup
import csv
from time import sleep
import random
import brotli

#2444页 重新跑一下
def get_pesticide_row(page_start = 1,page_end = 2444):
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
    for page in range(page_start, page_end+1):  # 2443页
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
                print(f"Page {page} Table not found")
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
                yield (djzh, nymc, nylb, jx, zhl, yxqz, cjmc)
                
            # 随机延迟防止封禁
            sleep(random.uniform(1, 3))

        except Exception as e:
            print(f"Page {page} Error: {str(e)}")
            continue


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
        # 写入CSV
        writer.writerow([djzh, nymc, nylb, jx, zhl, yxqz, cjmc])


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
    insert_sql = """INSERT INTO pesticide_overview (
    pesticide_regis_num, pesticide_name, pesticide_type, 
    dosage_form, total_content, validity_period, holder
) VALUES (%s, %s, %s, %s, %s, %s, %s)"""
    for idx,row in enumerate(get_pesticide_row(page_start=2445,page_end=2447)):
        try:
            cursor.execute(insert_sql, row)
         # 每20页提交一次事务（可调整）

            conn.commit()
        except Exception as e:
            print(f"Error inserting row {idx}: {e}")
    conn.commit()
    cursor.close()
    conn.close()