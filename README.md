# fucking_drug
Using large language models to generate pesticide formulas, this is my mentor's damn outsourced project

# 目录说明
## data_prepare文件夹
存放数据处理代码以及相应的数据
- `pesticide_regis_data`中国农药登记数据爬取,数据来源：[农药登记数据官网](http://www.icama.org.cn/zwb/dataCenter),处理结果放在了[huggingface](https://github.com/htesd/fucking_drug.git)上

## eval文件夹
模型评估代码
- `eval_bleu.py`根据模型生成结果和label计算bleu得分

## train文件夹
模型训练各阶段代码

test
