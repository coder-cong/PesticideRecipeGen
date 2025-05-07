# 读取目录下的jsonl文件并构建向量数据库
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from pymilvus import (MilvusClient,
                      DataType, 
                      Function, 
                      FunctionType,
                      FieldSchema,
                      CollectionSchema,
                      Collection,
                      connections,
                      utility)
import argparse
import random
from tqdm import tqdm
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

os.environ['http_proxy'] = "http://202.199.6.246:10809"
os.environ['https_proxy'] = "http://202.199.6.246:10809"


def save_to_database(documents,embeddings,client)->int:
    # 将文本存储到数据库中,返回下一个记录起始索引
    data = [
    { "vector": embeddings[i], "text": documents[i], }
    for i in range(len(embeddings))
    ]
    res = client.insert(collection_name=args.collection, data=data)
def read_jsonl(path:str="",filtered:list=[]):
    # 读取指定目录下的所有jsonl文件，yield返回一个文件的内容
    files = os.listdir(path)
    random.shuffle(files)
    print(files)
    data = []
    for file_name in files:
        # jsonl文件
        if ".jsonl" in file_name and file_name not in filtered:
            with open(os.path.join(path,file_name), 'r', encoding='utf-8') as f:
                for line in f:
                    if len(line.strip())!=0:
                        item = json.loads(line)
                        text = f"{item['instruction']}{item['input']}\n回答: {item['output']}"
                        data.append(text)
    return data

def encode_texts(documents,model):
    # 接受输入文本列表并编码成向量
    embeddings = model.encode(documents, convert_to_tensor=False, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    return embeddings
    

def init_client(dim):
    # Connect to Milvus given URI
    connections.connect(uri=args.server if args.server!="" else args.db_path)
    # 定义schema
    # Specify the data schema for the new Collection
    fields = [
        # Use auto generated id as primary key
        FieldSchema(
            name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
        ),
        # Store the original text to retrieve based on semantically distance
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        # Milvus now supports both sparse and dense vectors,
        # we can store each in a separate field to conduct hybrid search on both vectors
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields)
    
    # 新建collection
    if utility.has_collection(args.collection):
        Collection(args.collection).drop()
    client = Collection(args.collection, schema, consistency_level="Strong")

    # To make vector search efficient, we need to create indices for the vector fields
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    client.create_index("sparse_vector", sparse_index)
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    client.create_index("dense_vector", dense_index)
    client.load()
    
    return client

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理一个文件夹下面所有jsonl文件")
    parser.add_argument("--data_path",type=str,default="/home/liuxiaoming/clx/datasets/pesticide/proccessed",help="jsonl文件夹") 
    parser.add_argument("--model_path",type=str,default="/home/liuxiaoming/clx/models/SentenceTransformer_all_mpnet_base_v2",help="模型文件夹") 
    parser.add_argument("--db_path",type=str,default="./pesticide.db",help="存放数据库文件") 
    parser.add_argument("--server",type=str,default="",help="服务器ip和port")
    parser.add_argument("--collection",type=str,default="pesticide_data",help="表名") 
    args,unknown = parser.parse_known_args()
    # 加载模型
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cuda",cache_dir="/home/liuxiaoming/clx/models")
    dense_dim = ef.dim["dense"]
    #读取文件并编码
    texts = read_jsonl(args.data_path,["common_instruction.jsonl"])
    vectors = ef(texts)
    # 一次插入的长度
    sub_arr_len = 500

    # 创建client
    dimension = dense_dim
    client = init_client(dimension)
    
    # 遍历并插入到collection中
    for i in range(0, vectors["sparse"].shape[0], sub_arr_len):
        batched_entities = [
            texts[i : i + sub_arr_len],
            vectors["sparse"][i : i + sub_arr_len],
            vectors["dense"][i : i + sub_arr_len],
        ]
        client.insert(batched_entities)
    print("Number of entities inserted:", client.num_entities)