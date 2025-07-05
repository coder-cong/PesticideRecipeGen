# 在数据库连接配置下方添加查询函数
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

def get_all_pesticide_info(cursor):
    # 获取到所有农药的overview表中信息以及certificate表中的有效成分
    query = """
    SELECT
    t1.pesticide_regis_num,
    t1.pesticide_name,
    t1.pesticide_type,
    t1.dosage_form,
    t1.total_content,
    c1.active_components

FROM pesticide_overview t1
JOIN
#所有农药的有效成分
(
    SELECT
        pesticide_regis_num,
        GROUP_CONCAT(
            CONCAT(active_ingredient, ':', active_ingredient_content)
            ORDER BY active_ingredient SEPARATOR ';'
        ) AS active_components
    FROM pesticide_certificate
    GROUP BY pesticide_regis_num
) c1 ON t1.pesticide_regis_num = c1.pesticide_regis_num
    """

    cursor.execute(query)
    return cursor.fetchall()

if __name__ == "__main__":

    # ...（原有数据库连接和插入逻辑保持不变）
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
    all_list = []
    n_rows = 0
    all_regis_num = []
    all_list_no_group = []
    try:
        # 执行查询
        matching_pairs_data = get_matching_pairs(cursor)
        # 提交查询事务（如果需要）
        conn.commit()
        # 处理结果并检查每个登记证号列表长度
        for row in matching_pairs_data:
            pesticide_name = row[0]
            pesticide_type = row[1]
            dosage_form = row[2]
            total_content = row[3]
            active_components = row[4]
            matching_pairs_str = row[5]
            
            # 将字符串拆分为列表
            regis_nums = matching_pairs_str.split(',') if matching_pairs_str else []
            all_list.extend(regis_nums)
            n_rows += 1

        all_regis_num_res = get_all_regis_nums(cursor)
        conn.commit()
        for row in all_regis_num_res:
            all_regis_num.append(row[0])
            
        no_group_res = get_all_pesticide_info(cursor)
        conn.commit()
        for row in no_group_res:
            all_list_no_group.append(row[0])

    except Exception as e:
        print(f"查询或处理失败: {str(e)}")
        conn.rollback()  # 回滚事务

    finally:
        print(len(all_list),len(all_regis_num),n_rows)
        cursor.close()
        conn.close()
    for regis_num in all_list_no_group:
        if regis_num not in all_list:
            print(regis_num)
            break