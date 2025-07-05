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
from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
)

os.environ['http_proxy'] = "http://202.199.6.246:10809"
os.environ['https_proxy'] = "http://202.199.6.246:10809"

# milvus向量数据库语义搜索测试
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 混合搜索
def hybrid_search(
    col,
    query_dense_embedding,
    query_sparse_embedding,
    sparse_weight=1.0,
    dense_weight=1.0,
    limit=10,
):
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding.tolist()], "dense_vector", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [dense_req,sparse_req], rerank=rerank, limit=limit, output_fields=["text"]
    )[0]
    return [hit.get("text") for hit in res]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理一个文件夹下面所有jsonl文件")
    parser.add_argument("--data_path",type=str,default="/home/liuxiaoming/clx/datasets/pesticide/proccessed",help="jsonl文件夹") 
    parser.add_argument("--model_path",type=str,default="/home/liuxiaoming/clx/models/SentenceTransformer_all_mpnet_base_v2",help="模型文件夹") 
    parser.add_argument("--db_path",type=str,default="./pesticide.db",help="存放数据库文件") 
    parser.add_argument("--server",type=str,default="http://202.199.6.246:19530",help="服务器ip和port")
    parser.add_argument("--collection",type=str,default="pesticide_data",help="表名") 
    args,unknown = parser.parse_known_args()

    # 加载模型
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu",cache_dir="/home/liuxiaoming/clx/models")
    # 接受输入文本列表并编码成向量
    documents = ["Tebuconazole配方"]
    query_embeddings = ef(documents) 
    
    # 加载milvus客户端
    # Connect to Milvus given URI
    connections.connect(uri=args.db_path)
    col = Collection(args.collection)
    hybrid_results = hybrid_search(
    col,
    query_embeddings["dense"][0],
    query_embeddings["sparse"]._getrow(0),
    sparse_weight=1.0,
    dense_weight=0.8,
    limit=20,
)
    print(hybrid_results)