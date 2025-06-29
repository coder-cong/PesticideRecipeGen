import json
import os
import mysql.connector
from mysql.connector import Error
import uuid
from util.utils import connect_to_mysql


# 保存六种剂型需要的助剂到mysql数据库
def save_formulation_type_to_mysql(adjuvant_types, connection):
    """
    adjuvant_types为存储六种农药剂型所需要助剂的json文件
    """
    with open(adjuvant_types, "r", encoding="utf-8") as f:
        types = json.load(f)
        all_types = {}
        for type in types:
            adj_list = types[type]
            for adj in adj_list:
                if adj["name"] not in all_types:
                    all_types[adj["name"]] = adj["description"]
    print(all_types)
    cursor = connection.cursor()
    if connection is not None:
        # 将数据插入到adjuvant_info表中
        data_to_insert = [(str(uuid.uuid4()), name, description)
                          for name, description in all_types.items()]
        SQL = "INSERT INTO adjuvant_info(adj_id,adjuvant_name,adjuvant_description) VALUES (%s,%s,%s)"
        cursor.executemany(SQL, data_to_insert)
        connection.commit()
    else:
        raise ValueError("connection对象为空")


if __name__ == "__main__":
    connection = connect_to_mysql()
    save_formulation_type_to_mysql(
        "C:/Projs/PesticideRecipeGen/data/distill/type_adjuvant.json", connection)
