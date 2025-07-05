import json
import datasets
from datasets import load_dataset,Dataset
import os

def parse_json_list_to_jsonl(filename,encoding="utf-8"):
    with open(filename,"r",encoding=encoding) as f:
        data = json.load(f)
        ds = Dataset.from_list(data)
        ds.to_json(os.path.splitext(filename)[0]+".jsonl",force_ascii=False)
        
        
if __name__ == "__main__":
    parse_json_list_to_jsonl("C:\\Users\\Cong\\Desktop\\myProj\\PesticideRecipeGen\\data_prepare\\pretrain_data\process_out\\专利_split.json")
    parse_json_list_to_jsonl("C:\\Users\\Cong\\Desktop\\myProj\\PesticideRecipeGen\\data_prepare\\pretrain_data\\process_out\\农药制剂学_filtered.json")