"""
工具函数
"""
from typing import Any, Optional
import re
import json
import mysql.connector
from mysql.connector import Error

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'mysql123',
    'database': 'pesticide'
}


def connect_to_mysql():
    """
    使用db_config连接到数据库并获取到cursor
    """
    connection = mysql.connector.connect(**db_config)
    return connection


def extract_json_from_markdown(text: str) -> Optional[str]:
    """
    从包含Markdown代码块的文本中提取纯净的JSON字符串。
    该函数会查找被 ```json ... ``` 或 ``` ... ``` 包裹的内容。
    Args:
        text (str): 可能包含JSON代码块的原始字符串。
    Returns:
        Optional[str]: 如果找到，返回提取并清理过的JSON字符串；否则返回None。
    """
    # 正则表达式模式，用于匹配被```json ...```或``` ...```包裹的内容。
    # re.DOTALL 标志让 `.` 可以匹配包括换行符在内的任何字符。
    # (?:json)? 表示 "json" 这个词是可选的，并且我们不捕获它。
    # \s* 用来匹配代码块标记后可能存在的任何空格或换行。
    # (.*?) 是一个非贪婪的捕获组，用于捕获我们需要的JSON内容。
    pattern = r"```(?:json)?\s*(.*?)\s*```"

    match = re.search(pattern, text, re.DOTALL)

    if match:
        # group(1) 返回第一个捕获组的内容，即括号中的 (.*?) 部分
        # .strip() 用于移除可能存在于JSON内容前后的多余空格或换行
        return match.group(1).strip()

    return None


def extract_and_parse_json(text: str) -> Optional[Any]:
    """
    从包含Markdown代码块的文本中提取JSON字符串，并将其解析为Python对象。
    首先调用 extract_json_from_markdown 提取字符串，然后尝试用 json.loads() 解析。
    Args:
        text (str): 可能包含JSON代码块的原始字符串。
    Returns:
        Optional[Any]: 如果成功提取并解析，返回Python对象（通常是dict或list）；
                       如果在任何步骤失败（未找到、或内容不是有效JSON），则返回None。
    """
    json_string = extract_json_from_markdown(text)

    if json_string:
        try:
            # 尝试将提取出的字符串解析为Python对象
            return json.loads(json_string)
        except json.JSONDecodeError:
            # 如果字符串不是有效的JSON格式，解析会失败
            print("警告：找到了JSON代码块，但其内容不是有效的JSON格式。")
            return None

    return None
