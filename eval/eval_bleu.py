from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
import os
import random
from datasets import load_dataset
import torch
import json
import sacrebleu
import torch.nn.functional as F
from tqdm import tqdm

# 在训练数据集上抽取数据进行推理，计算模型推理结果和label的bleu得分

# 模型和lora路径
model_path = "/home/liuxiaoming/clx/models/qwen2.5_14B_r1_distill"
tokenizer_path = "/home/liuxiaoming/clx/models/qwen2.5_14B_instruct"
adapter_path = "/home/liuxiaoming/clx/projs/LLaMA-Factory/src/saves/Qwen2.5-14B-Instruct/lora/qwen14B-r1-lora48-dropout0.1-epoch3-neftune0.1/checkpoint-20000"

#cuda设备
device = "cuda"

#数据集路径
data_path = "/home/liuxiaoming/clx/datasets/pesticide/proccessed"

# 生成预测结果
def generate(inputs,model_used,max_length=1024,temperature=0.3):
    # inputs为instruction列表
    # 批量应用模板
    messages = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "你是一个东北大学太一组研发的农药专家，擅长解答农药领域相关问题"},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": ""},
            ],
            return_tensors="pt",
        ).to(device)[:, :-2]
        for input_text in inputs
    ]
    msg_max_len = max(message.shape[-1] for message in messages) 
    padded_messages = [F.pad(msg, (msg_max_len-msg.shape[-1], 0,0,0), mode='constant', value=tokenizer.pad_token_id) for msg in messages]
    padded_inputs = torch.cat(padded_messages,dim=0)
     # 生成 attention mask
    attention_mask = (padded_inputs != tokenizer.pad_token_id).long()
    
    # 批量推理
    outputs = model_used.generate(
        input_ids=padded_inputs,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=None,
        do_sample=False,
        top_p = None,
    )
    
    # 解码并忽略特殊 Token
    responses = []
    for i, output in enumerate(outputs):
        response_start = padded_messages[i].shape[-1]
        response = output[response_start:]
        response_text = tokenizer.decode(response, skip_special_tokens=True)
        responses.append(response_text)

    return responses

if __name__=="__main__":
    # 加载模型和适配器
    base_model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    base_model.to(device)
    
    model = PeftModel.from_pretrained(base_model,adapter_path)
    
    base_eval_list = []
    sft_eval_list = []
    
    # 读取数据集并进行评估
    for file in tqdm(os.listdir(data_path)):
        # 读取处理jsonl文件
        if "jsonl" in file:
            ds = load_dataset("json",data_files=os.path.join(data_path,file))['train']
            # 检查数据量并随机抽取30条进行生成
            sample_size = 20
            if len(ds) < sample_size:
                raise ValueError("数据集中的数据不足30条")
            random_indices = random.sample(range(len(ds)), sample_size)
            sampled_data = ds.select(random_indices)
            # # 使用base_model进行推理
            # gene_res = generate(sampled_data['instruction'],base_model)
            # base_scores = []
            # # 计算bleu得分
            # for index,(predict,label) in enumerate(zip(gene_res,sampled_data["output"])):
            #     bleu = sacrebleu.sentence_bleu(predict, [label]).score
            #     base_scores.append((index,bleu))
            # sorted_base_scores = sorted(base_scores,key=lambda x:x[1])
            # # 保存倒数5个以及前五个
            # for (index,bleu) in sorted_base_scores[:5]+sorted_base_scores[-5:]:
            #     eval_res = {
            #         "dataset":file.split(".")[0],
            #         "bleu":bleu,
            #         "instruction":sampled_data["instruction"][index],
            #         "predict":gene_res[index],
            #         "label":sampled_data["output"][index]
            #     }
            #     base_eval_list.append(eval_res)
            # 使用sft后的model进行推理
            gene_res = generate(sampled_data['instruction'],model)
            scores = []
            # 计算bleu得分
            for index,(predict,label) in enumerate(zip(gene_res,sampled_data["output"])):
                bleu = sacrebleu.sentence_bleu(predict, [label]).score
                scores.append((index,bleu))
            sorted_scores = sorted(scores,key=lambda x:x[1])
            # 保存倒数5个与前五个
            for (index,bleu) in sorted_scores[:5]+sorted_scores[-5:]:
                eval_res = {
                    "dataset":file.split(".")[0],
                    "bleu":bleu,
                    "instruction":sampled_data["instruction"][index],
                    "predict":gene_res[index],
                    "label":sampled_data["output"][index]
                }
                sft_eval_list.append(eval_res)
    # with open("base_result.json","w",encoding="utf-8") as f:
    #     json.dump(base_eval_list,f,ensure_ascii=False,indent=4)        
    with open("sft_result.json","w",encoding="utf-8") as f:
        json.dump(sft_eval_list,f,ensure_ascii=False,indent=4) 

    