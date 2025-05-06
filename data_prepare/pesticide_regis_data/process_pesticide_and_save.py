"""
数据库中查询农药数据并保存到jsonl中
"""
from tqdm import tqdm
# 筛选出所有农药名称、剂型、总含量、有效成分以及含量都相同的农药
def get_matching_pairs(cursor):
    query = """
    SELECT 
        t1.pesticide_name,
        t1.pesticide_type,
        t1.dosage_form,
        t1.total_content,
        c1.active_components,
        GROUP_CONCAT(DISTINCT t1.pesticide_regis_num 
            ORDER BY t1.pesticide_regis_num SEPARATOR ',') AS matching_pairs
    FROM pesticide_overview t1
    JOIN (
        SELECT 
            pesticide_regis_num,
            GROUP_CONCAT(
                CONCAT(active_ingredient, ':', active_ingredient_content) 
                ORDER BY active_ingredient SEPARATOR ';' 
            ) AS active_components
        FROM pesticide_certificate
        GROUP BY pesticide_regis_num
    ) c1 
    ON t1.pesticide_regis_num = c1.pesticide_regis_num
    GROUP BY 
        t1.pesticide_name, 
        t1.pesticide_type, 
        t1.dosage_form, 
        t1.total_content, 
        c1.active_components;
    """
    cursor.execute(query)
    return cursor.fetchall()


def get_all_regis_nums(cursor):
    query = """
    SELECT pesticide_regis_num 
    FROM pesticide_overview 
    """
    cursor.execute(query)
    return cursor.fetchall()


# 根据注册号查询label的数据
def query_label_by_regis_num(cursor, regis_num):
    query = """
    SELECT * FROM pesticide_label WHERE pesticide_regis_num = %s
    """
    cursor.execute(query, (regis_num,))
    return cursor.fetchall()


# 根据注册号查询certificate的数据
def query_certificate_by_regis_num(cursor, regis_num):
    query = """
    SELECT * FROM pesticide_certificate WHERE pesticide_regis_num = %s
    """
    cursor.execute(query, (regis_num,))
    return cursor.fetchall()


# 根据注册号查询overview表中数据
def query_overview_by_regis_num(cursor, regis_num):
    query = """
    SELECT * FROM pesticide_overview WHERE pesticide_regis_num = %s
    """
    cursor.execute(query, (regis_num,))
    return cursor.fetchall()


if __name__ == "__main__":

    # ...（原有数据库连接和插入逻辑保持不变）
    import pymysql

    # 在基础配置后新增数据库配置
    db_config = {
        "host": "localhost",
        "port": 3306,
        "user": "root",
        "password": "12345",
        "database": "pesticide_data",
        "charset": "utf8mb4",
    }
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()
    row_num = 0
    # 筛选出所有农药名称、剂型、总含量、有效成分以及含量都相同的农药，对于这些农药只保留一个到数据集中
    total_data = []
    label_info_weights = {
        "scope_of_use_and_usage": 0.3,
        "product_performance": 0.2,
        "things_to_note": 0.2,
        "first_aid_measures": 0.2,
        "storage_and_transportation_methods": 0.1,
    }
    try:
        # 执行查询
        matching_pairs_data = get_matching_pairs(cursor)
        # 提交查询事务（如果需要）
        conn.commit()
        # 处理结果并检查每个登记证号列表长度
        for row in tqdm(matching_pairs_data, desc="Processing Rows"):
            data = {
                "pesticide_name": "",
                "pesticide_type": "",
                "dosage_form": "",
                "total_content": "",
                "toxicity": "",
                # 多个有效成分以分号进行分隔
                "active_ingredients": "",
                "active_ingredient_english_names": "",
                "active_ingredient_contents": "",
                #label中信息
                "scope_of_use_and_usage": "",
                "product_performance": "",
                "things_to_note": "",
                "first_aid_measures": "",
                "storage_and_transportation_methods": "",
            }

            data["pesticide_name"] = row[0]
            data["pesticide_type"] = row[1]
            data["dosage_form"] = row[2]
            data["total_content"] = row[3]
            matching_pairs_str = row[5]

            # 将字符串拆分为列表
            regis_nums = matching_pairs_str.split(",") if matching_pairs_str else []

            # 遍历农药名字、类型、剂型、总含量、有效成分和含量都相同的农药注册号，查询label数据，选择加权长度最长的农药数据
            max_len = 0
            sel_regis_num = ""
            scope_of_use_and_usage = ""
            product_performance = ""
            things_to_note = ""
            first_aid_measures = ""
            storage_and_transportation_methods = ""
            
            for regis_num in regis_nums:
                label_data = query_label_by_regis_num(cursor, regis_num)
                len_scope_of_use_and_usage = len(label_data[0][1]) if label_data else 0
                len_product_performance = len(label_data[0][2]) if label_data else 0
                len_things_to_note = len(label_data[0][3]) if label_data else 0
                len_first_aid_measures = len(label_data[0][4]) if label_data else 0
                len_storage_and_transportation_methods = (
                    len(label_data[0][5]) if label_data else 0
                )
                reg_len = (
                    len_scope_of_use_and_usage
                    * label_info_weights["scope_of_use_and_usage"]
                    + len_product_performance
                    * label_info_weights["product_performance"]
                    + len_things_to_note * label_info_weights["things_to_note"]
                    + len_first_aid_measures * label_info_weights["first_aid_measures"]
                    + len_storage_and_transportation_methods
                    * label_info_weights["storage_and_transportation_methods"]
                )
                if max_len <= reg_len:
                    max_len = reg_len
                    sel_regis_num = regis_num
                    scope_of_use_and_usage = label_data[0][1] if label_data else ""
                    product_performance = label_data[0][2] if label_data else ""
                    things_to_note = label_data[0][3] if label_data else ""
                    first_aid_measures = label_data[0][4] if label_data else ""
                    storage_and_transportation_methods = (
                        label_data[0][5] if label_data else ""
                    )
            data["scope_of_use_and_usage"] = scope_of_use_and_usage
            data["product_performance"] = product_performance
            data["things_to_note"] = things_to_note
            data["first_aid_measures"] = first_aid_measures
            data["storage_and_transportation_methods"] = storage_and_transportation_methods

            # 检查总行数
            if sel_regis_num is  None:
                print(regis_nums)

            # 获取毒性信息
            pesticide_table_row = query_overview_by_regis_num(cursor, sel_regis_num)
            data["toxicity"] = pesticide_table_row[0][7] if pesticide_table_row else ""

            # 获取拼接有效成分信息
            active_components = query_certificate_by_regis_num(cursor, sel_regis_num)
            for i, row in enumerate(active_components):
                data["active_ingredients"] += row[1]
                data["active_ingredient_english_names"] += row[2]
                data["active_ingredient_contents"] += row[3]
                if i != len(active_components) - 1:
                    data["active_ingredients"] += ";"
                    data["active_ingredient_english_names"] += ";"
                    data["active_ingredient_contents"] += ";"
            
            #将农药数据追加到所有数据末尾
            total_data.append(data)
    except Exception as e:
        print(f"查询或处理失败: {str(e)}")
        conn.rollback()  # 回滚事务

    finally:
        print(len(total_data))
        import datasets
        dataset = datasets.Dataset.from_list(total_data)
        dataset.to_json("pesticide_data.jsonl",force_ascii=False)
