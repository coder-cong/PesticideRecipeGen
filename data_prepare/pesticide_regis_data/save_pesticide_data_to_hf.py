"""
将农药剂型以及农药注册数据保存到huggingface中
"""

import os
os.environ['http_proxy'] = "http://202.199.6.246:10809"
os.environ['https_proxy'] = "http://202.199.6.246:10809"

dosages = [
    {
    "dosage_form": "原药",
    "dosage_english_name": "technical material",
    "dosage_code": "TC",
    "dosage_info": "在制造过程中得到有效成分及有关组分组成的产品，必要时可加入少量的添加剂（稳定剂）。"
  },
  {
    "dosage_form": "母药",
    "dosage_english_name": "technical concentrate",
    "dosage_code": "TK",
    "dosage_info": "在制造过程中得到有效成分及有关组分组成的产品，可能含有少量必需的添加剂（如稳定剂）和适当的稀释剂。"
  },
  {
    "dosage_form": "粉剂",
    "dosage_english_name": "dustable powder",
    "dosage_code": "DP",
    "dosage_info": "适用喷粉或撒布含有效成分的自由流动粉状制剂。"
  },
  {
    "dosage_form": "颗粒剂",
    "dosage_english_name": "granule",
    "dosage_code": "GR",
    "dosage_info": "具有一定粒径范围可自由流动含有效成分的粒状制剂。"
  },
  {
    "dosage_form": "球剂",
    "dosage_english_name": "pellet",
    "dosage_code": "PT",
    "dosage_info": "含有效成分的球状制剂（一般直径大于6 mm）。"
  },
  {
    "dosage_form": "片剂",
    "dosage_english_name": "tablet",
    "dosage_code": "TB",
    "dosage_info": "具有一定形状和大小含有效成分的片状制剂（通常具有两平面或凸面，两面间距离小于直径）。"
  },
  {
    "dosage_form": "条剂",
    "dosage_english_name": "plant rodlet",
    "dosage_code": "PR",
    "dosage_info": "含有效成分的条状或棒状制剂（一般长为几厘米，宽度/直径几毫米，即长度大于直径/宽度）。"
  },
{"dosage_form":"可湿性粉剂","dosage_english_name":"wettable powder","dosage_code":"WP","dosage_info":"有效成分在水中分散成悬浮液的粉状制剂。"},
{"dosage_form":"油分散粉剂","dosage_english_name":"oil dispersible powder","dosage_code":"OP","dosage_info":"有效成分在有机溶剂中分散成悬浮液的粉状制剂。"},
{"dosage_form":"乳粉剂","dosage_english_name":"emulsifiable powder","dosage_code":"EP","dosage_info":"有效成分被有机溶剂溶解、包裹在可溶或不溶的惰性成分中，在水中分散形成水包油乳液的粉状制剂。"},
{"dosage_form":"水分散粒剂","dosage_english_name":"water dispersible granule","dosage_code":"WG","dosage_info":"在水中崩解、有效成分分散成悬浮液的粒状制剂。"},
{"dosage_form":"乳粒剂","dosage_english_name":"emulsifiable granule","dosage_code":"EG","dosage_info":"有效成分被有机溶剂溶解、包裹在可溶或不溶的惰性成分中，在水中分散形成水包油乳液的粒状制剂。"},
{"dosage_form":"水分散片剂","dosage_english_name":"water dispersible tablet","dosage_code":"WT","dosage_info":"在水中崩解、有效成分分散成悬浮液的片状制剂。"},
{"dosage_form":"可溶粉剂","dosage_english_name":"water soluble powder","dosage_code":"SP","dosage_info":"有效成分在水中形成真溶液的粉状制剂，可含有不溶于水的惰性成分。"},
{"dosage_form":"可溶粒剂","dosage_english_name":"water soluble granule","dosage_code":"SG","dosage_info":"有效成分在水中形成真溶液的粒状制剂，可含不溶于水的惰性成分。"},
{"dosage_form":"可溶片剂","dosage_english_name":"water soluble tablet","dosage_code":"ST","dosage_info":"有效成分在水中形成真溶液的片状制剂，可含不溶于水的惰性成分。"},
{"dosage_form":"可溶液剂","dosage_english_name":"soluble concentrate","dosage_code":"SL","dosage_info":"用水稀释成透明或半透明含有效成分的液体制剂，可含有不溶于水的惰性成分。"},
{"dosage_form":"可溶胶剂","dosage_english_name":"water soluble gel","dosage_code":"GW","dosage_info":"用水稀释成真溶液含有效成分的胶状制剂。"},
{"dosage_form":"油剂","dosage_english_name":"oil miscible liquid","dosage_code":"OL","dosage_info":"用有机溶剂稀释 (或不稀释) 成均相、含有效成分的液体制剂。"},
{"dosage_form":"展膜油剂","dosage_english_name":"spreading oil","dosage_code":"SO","dosage_info":"在水面自动扩散成油膜含有效成分的油剂。"},
{"dosage_form":"乳油","dosage_english_name":"emulsifiable concentrate","dosage_code":"EC","dosage_info":"用水稀释分散成乳状液含有效成分的均相液体制剂。"},
{"dosage_form":"乳胶","dosage_english_name":"emulsifiable gel","dosage_code":"GL","dosage_info":"用水稀释分散成乳状液含有效成分的乳胶制剂。"},
{"dosage_form":"可分散液剂","dosage_english_name":"dispersible concentrate","dosage_code":"DC","dosage_info":"用水稀释分散成悬浮含有效成分的均相液体制剂。"},
{"dosage_form":"膏剂","dosage_english_name":"paste","dosage_code":"PA","dosage_info":"含有效成分可成膜的水基膏状制剂，一般直接使用。"},
{"dosage_form":"水乳剂","dosage_english_name":"emulsion,oil in water","dosage_code":"EW","dosage_info":"有效成分 (或其有机溶液) 在水中形成乳状液体制剂。"},
{"dosage_form":"油乳剂","dosage_english_name":"emulsion,water in oil","dosage_code":"EO","dosage_info":"有效成分 (或其水溶液) 在油中形成乳状液体制剂。"},
{"dosage_form":"微乳剂","dosage_english_name":"micro- emulsion","dosage_code":"ME","dosage_info":"有效成分在水中成透明或半透明的微乳状液体制剂，直接或用水稀释后使用。"},
{"dosage_form":"脂剂","dosage_english_name":"grease","dosage_code":"GS","dosage_info":"含有效成分的油或脂肪基黏稠制剂，一般直接使用。"},
{"dosage_form":"悬浮剂","dosage_english_name":"suspension concen- trate","dosage_code":"SC","dosage_info":"有效成分以固体微粒分散在水中成稳定的悬浮液体制剂，一般用水稀释使用。"},
{"dosage_form":"微囊悬浮剂","dosage_english_name":"capsule suspension","dosage_code":"CS","dosage_info":"含有效成分的微囊分散在液体中形成稳定的悬浮液体制剂。"},
{"dosage_form":"油悬浮剂","dosage_english_name":"oil miscible flowable concentrate","dosage_code":"OF","dosage_info":"有效成分以固体微粒分散在液体中成稳定的悬浮液体制剂，一般用有机溶剂稀释使用。"},
{"dosage_form":"可分散油悬浮剂","dosage_english_name":"oil-based suspension concentrate（oil dis- persion）","dosage_code":"OD","dosage_info":"有效成分以固体微粒分散在非水介质中成稳定的悬浮液体制剂，一般用水稀释使用。"},
{"dosage_form":"悬乳剂","dosage_english_name":"suspo-emulsion","dosage_code":"SE","dosage_info":"有效成分以固体微粒和水不溶的微小液滴形态稳定分散在连续的水相中成非均相液体制剂。"},
{"dosage_form":"微囊悬浮 - 悬浮剂","dosage_english_name":"mixed formulations of CS and SC","dosage_code":"ZC","dosage_info":"有效成分以微囊及固体微粒分散在水中成稳定的悬浮液体制剂。"},
{"dosage_form":"微囊悬浮 - 水乳剂","dosage_english_name":"mixed formulations of CS and EW","dosage_code":"ZW","dosage_info":"有效成分以微囊、微小液滴形态稳定分散在连续的水相中成非均相液体制剂。"},
{"dosage_form":"微囊悬浮 - 悬乳剂","dosage_english_name":"mixed formulations of CS and SE","dosage_code":"ZE","dosage_info":"有效成分以微囊、固体颗粒和微小液滴形态稳定分散在连续的水相中成非均相液体制剂。"},
{"dosage_form":"种子处理干粉剂","dosage_english_name":"powder for dry seed treatment","dosage_code":"DS","dosage_info":"直接用于种子处理含有效成分的干粉制剂。"},
{"dosage_form":"种子处理可分散粉剂","dosage_english_name":"water dispersible powder for slurry seed treatment","dosage_code":"WS","dosage_info":"用水分散成高浓度浆状含有效成分的种子处理粉状制剂。"},
{"dosage_form":"种子处理液剂","dosage_english_name":"solution for seed treatment","dosage_code":"LS","dosage_info":"直接或稀释用于种子处理含有效成分、透明或半透明的液体制剂，可能含有不溶水的惰性成分。"},
{"dosage_form":"种子处理乳剂","dosage_english_name":"emulsion for seed treatment","dosage_code":"ES","dosage_info":"直接或稀释用于种子处理含有效成分、稳定的乳液制剂。"},
{"dosage_form":"种子处理悬浮剂","dosage_english_name":"suspension concentrate for seed treatment (flo-wable concentrate for seed treatment)","dosage_code":"FS","dosage_info":"直接或稀释用于种子处理含有效成分、稳定的悬浮液体制剂。"},
{"dosage_form":"气雾剂","dosage_english_name":"aerosol dispenser","dosage_code":"AE","dosage_info":"按动阀门在抛射剂作用下，喷出含有效成分药液的微小液珠或雾滴的密封罐装制剂。"},
{"dosage_form":"电热蚊香片","dosage_english_name":"vaporizing mat","dosage_code":"MV","dosage_info":"以纸片或其他为载体，在配套加热器加热，使有效成分挥发的片状制剂。"},
{"dosage_form":"电热蚊香液","dosage_english_name":"liquid vaporizer","dosage_code":"LV","dosage_info":"在盛药液瓶与配套加热器配合下，通过加热器芯棒使有效成分挥发的均相液体制剂。"},
{"dosage_form":"防蚊片","dosage_english_name":"proof mat","dosage_code":"PM","dosage_info":"以合成树脂或其他为载体，在配套风扇等的风力作用下，使有效成分挥发的片状或粒状制剂。"},
{"dosage_form":"气体制剂","dosage_english_name":"gas","dosage_code":"GA","dosage_info":"有效成分在耐压容器内压缩的气体制剂。"},
{"dosage_form":"发气剂","dosage_english_name":"gas generating product","dosage_code":"GE","dosage_info":"以化学反应产生有效成分的气体制剂。"},
{"dosage_form":"挥散芯","dosage_english_name":"dispensor","dosage_code":"DR","dosage_info":"利用载体释放有效成分，用于调控昆虫行为的制剂。"},
{"dosage_form":"烟剂","dosage_english_name":"smoke generator","dosage_code":"FU","dosage_info":"通过点燃发烟 (或经化学反应产生的热能) 释放有效成分的固体制剂。"},
{"dosage_form":"蚊香","dosage_english_name":"mosquito coil","dosage_code":"MC","dosage_info":"点燃 (熏烧) 后不会产生明火，通过烟将有效成分释放到空间的螺旋形盘状制剂。"},
{"dosage_form":"饵剂","dosage_english_name":"bait (ready for use)","dosage_code":"RB","dosage_info":"为引诱靶标有害生物取食直接使用、含有有效成分的制剂。"},
{"dosage_form":"浓饵剂","dosage_english_name":"bait concentrate","dosage_code":"CB","dosage_info":"稀释后使用、含有有效成分的固体或液体饵剂。"},
{"dosage_form":"防蚊网","dosage_english_name":"insect-proof net","dosage_code":"PN","dosage_info":"以合成树脂或其他为载体，释放有效成分的网状制剂。"},
{"dosage_form":"防虫罩","dosage_english_name":"insect-proof cover","dosage_code":"PC","dosage_info":"以无纺布或其他为载体，释放有效成分的网状制剂。"},
{"dosage_form":"长效防蚊帐","dosage_english_name":"long-lasting insecticidal net","dosage_code":"LN","dosage_info":"以合成纤维或其他为载体，释放有效成分，以物理和化学屏障防治害虫的蚊帐制剂。"},
{"dosage_form":"驱蚊乳","dosage_english_name":"repellent milk","dosage_code":"RK","dosage_info":"直接涂抹皮肤，具有驱避作用、含有有效成分的乳液制剂。"},
{"dosage_form":"驱蚊液","dosage_english_name":"repellent liquid","dosage_code":"RQ","dosage_info":"直接涂抹皮肤，具有驱避作用、含有有效成分或可有黏度的清澈液体制剂。"},
{"dosage_form":"驱蚊花露水","dosage_english_name":"repellent floral water","dosage_code":"RW","dosage_info":"直接涂抹皮肤，具有驱避作用、含有有效成分的清澈花露水液体制剂。"},
{"dosage_form":"驱蚊巾","dosage_english_name":"repellent wipe","dosage_code":"RP","dosage_info":"直接擦抹皮肤，具有驱避作用、含有效成分药液的湿无纺布或其他载体制剂。"},
{"dosage_form":"超低容量液剂","dosage_english_name":"ultra low volume liquid","dosage_code":"UL","dosage_info":"直接或稀释后在超低容量设备上使用的均相液体制剂。"},
{"dosage_form":"热雾剂","dosage_english_name":"hot fogging concentrate","dosage_code":"HN","dosage_info":"直接或稀释后在热雾设备上使用的制剂。"}
]

import datasets
from datasets import load_dataset,DatasetDict,Dataset

# 确保所有条目字段一致（如dosage_code为空的条目需处理）
dataset = datasets.Dataset.from_list(dosages)
print(dataset)

from huggingface_hub import HfApi

from huggingface_hub import login

login(token="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")

# 创建DatasetDict并推送
dataset_dict = datasets.DatasetDict({"train": dataset})
dataset_dict.push_to_hub("player0001/pesticide_dosage_form")


"""
保存农药注册数据
"""
# 加载数据并推送到Hugging Face Hub（需要使用ctrl+f5运行否则不会加载所有数据）
dataset_stream = load_dataset(
    "json",
    data_files="pesticide_data.jsonl",
)
len = 0
for data in dataset_stream['train']:
    len += 1
print(len)
# dataset_stream.push_to_hub("player0001/pesticide_registration_data")
