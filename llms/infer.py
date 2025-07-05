from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/media/iiap/25df545d-3a24-4466-b58d-f96c46b9a3bf/LargeModel/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "斯奈德夫人过去将每月收入的 40% 用于房租和水电费。她的工资最近增加了 600 美元，所以现在她的租金和水电费只占她月收入的 25%。她之前的月收入是多少？"
messages = [
    {"role": "system", "content": "你是qwen，请你认真思考问题并且回答用户的需求。"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


print(response)