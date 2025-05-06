import requests
from bs4 import BeautifulSoup
import csv
from time import sleep
import random
import brotli
import re
import math
import random

#TODO 305 306 668 669 1294页错误，2444页 重新跑一下
#爬取https://www.icama.cn/BasicdataSystem/pesticideRegistration/queryselect.do页面的table
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
        print(f"Page {page} Processing...")
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

                a_tag = cols[0].find("a")
                if a_tag and (onclick_str := a_tag.get('onclick')):
                    viewpd_id = re.search(r"_viewpd\('(.*?)'\)", onclick_str).group(1)
                else :
                    viewpd_id = ""
                    
                data = {
                    "r": f"{random.random()}",
                    "id" : viewpd_id,
                    f"___t{random.random()}":""
                }

                certificate_url = "https://www.icama.cn/BasicdataSystem/pesticideRegistration/viewpd.do"
                # 带重试机制的请求
                for _ in range(3):  # 重试3次
                    response = session.post(
                    certificate_url,
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
                
                certificate_data = parse_certificate_detail(response.text)

                yield (page , djzh, certificate_data)
                
            # 随机延迟防止封禁
            sleep(random.uniform(0, 0.1))

        except Exception as e:
            print(f"Page {page} Failed: {str(e)}")
            continue

def parse_certificate_detail(html_content):
    """解析证书详情页面的有效成分和毒性信息"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 初始化结果字典
    result = {
        "毒性": "",
        "有效成分": []
    }

    # 解析毒性信息（位于第一个登记证表格）
    for table in soup.find_all('table', id='reg'):
        if '农药登记证信息' in table.get_text():
            toxicity_td = table.find('td', string='毒性：')
            if toxicity_td:
                result["毒性"] = toxicity_td.find_next_sibling('td').get_text(strip=True)
            break

    # 解析有效成分表格
    active_table = None
    for table in soup.find_all('table', id='reg'):
        if '有效成分信息' in table.get_text():
            active_table = table
            break

    if active_table:
        # 跳过标题行（前两行：表格标题和列标题）
        for row in active_table.find_all('tr')[2:]:
            cols = row.find_all('td')
            if len(cols) >= 3:
                result["有效成分"].append({
                    "成分名称": cols[0].get_text(strip=True),
                    "英文名称": cols[1].get_text(strip=True),
                    "含量": cols[2].get_text(strip=True)
                })

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
        # 写入CSV
        writer.writerow([djzh, nymc, nylb, jx, zhl, yxqz, cjmc])
        
#数据库配置
db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '12345',
    'database': 'pesticide_data',
    'charset': 'utf8mb4'
    }

if __name__ == "__main__":
    import pymysql
    # 在基础配置后新增数据库配置

    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    # 修改后的SQL语句
    update_overview_sql = """UPDATE pesticide_overview 
                            SET toxicity = %s 
                            WHERE pesticide_regis_num = %s"""

    insert_certificate_sql = """INSERT INTO pesticide_certificate 
                              (pesticide_regis_num, active_ingredient, 
                              active_ingredient_english_name, active_ingredient_content)
                              SELECT %s, %s, %s, %s
                              FROM DUAL 
                              WHERE NOT EXISTS (
                                  SELECT 1 
                                  FROM pesticide_certificate 
                                  WHERE pesticide_regis_num = %s 
                                    AND active_ingredient = %s 
                                    AND active_ingredient_english_name = %s 
                                    AND active_ingredient_content = %s
                              )"""

#   1135页需要重新跑一下 1254开始
    for idx, row_data in enumerate(get_pesticide_row(page_start=1, page_end=2447)):
        try:
            # 解包返回数据
            page, djzh, detail_data = row_data
            
            # 更新overview表的毒性字段 更新完成
            cursor.execute(update_overview_sql,
                          (detail_data["毒性"], djzh))
            
            # # 插入certificate表数据
            for ingredient in detail_data["有效成分"]:
                # 执行时需要传递两次参数（注意顺序匹配）
                cursor.execute(insert_certificate_sql,
               (djzh, ingredient["成分名称"], ingredient["英文名称"], ingredient["含量"],
                djzh, ingredient["成分名称"], ingredient["英文名称"], ingredient["含量"]))
            
            # 每20次操作提交一次

            conn.commit()
                
        except Exception as e:
            print(f"Processing Pesticide{djzh} Failed(MYSQL): {str(e)}")
            conn.rollback()  # 回滚当前事务        
            
    conn.commit()
    cursor.close()
    conn.close()