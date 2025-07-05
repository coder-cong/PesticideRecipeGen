import os
import random
import time
import json
import numpy as np
import torch
import cv2
import os.path as path
from torch.utils.data.dataset import Dataset
from torchvision import datasets
import torchvision.transforms as T
from PIL import Image
import jsonlines
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import os
import skimage.io as io

'''
对于gan网络而言，单次的输入随机变量也是很重要的。这里我先改为一个一维变量尝试下

'''
def sort_strings(strings):
    sorted_strings = sorted(strings)
    string_dict = {string: index for index, string in enumerate(sorted_strings)}
    return string_dict


TEST=False

data_location = "../data"


def num_to_onehot(num,total_num):

    ont_hot=torch.zeros([total_num])
    ont_hot[num]=1
    return ont_hot

#                                    T.Normalize(0.1370, 0.3081)
def get_minist_dataset(name="mnist"):
    assert name in ['mnist']
    if name == 'mnist':
        train = datasets.MNIST(root=data_location,
                               download=True,
                               transform=T.Compose([

                                   T.ToTensor(),

                               ]),
                               train=True)
        test = datasets.MNIST(root=data_location,
                              download=True,
                              transform=T.Compose([

                                  T.ToTensor(),

                              ]),
                              train=False)
    return train, test


def get_fashion_minist_dataset(name="fashion_mnist"):
    assert name in ['fashion_mnist']
    if name == 'fashion_mnist':
        train = datasets.FashionMNIST(root=data_location,
                               download=True,
                               transform=T.Compose([

                                   T.ToTensor(),

                               ]),
                               train=True)
        test = datasets.FashionMNIST(root=data_location,
                              download=True,
                              transform=T.Compose([

                                  T.ToTensor(),

                              ]),
                              train=False)
    return train, test


def get_cifar10(name="fashion_mnist"):
    assert name in ['fashion_mnist']
    if name == 'fashion_mnist':
        train = datasets.CIFAR10(root=data_location,
                               download=True,
                               transform=T.Compose([

                                   T.ToTensor(),

                               ]),
                               train=True)
        test = datasets.CIFAR10(root=data_location,
                              download=True,
                              transform=T.Compose([

                                  T.ToTensor(),

                              ]),
                              train=False)
    return train, test


class dataset_from_image_folder(Dataset):
    def __init__(self, image_dir, transform=None):
        self.trans = transform

        self.image_list = []
        self.label_list = []

        if os.path.isdir(image_dir):
            sorted_string_dict = sort_strings(os.listdir(image_dir))
            self.label_num = len(sorted_string_dict)

            for label, dir_name in enumerate(sorted_string_dict):
                file_list = os.listdir(os.path.join(image_dir, dir_name))
                for file_name in file_list:
                    file_path = os.path.join(image_dir, dir_name, file_name)
                    if os.path.isfile(file_path):
                        self.image_list.append(file_path)
                        self.label_list.append(label)
        else:
            print("请检查文件路径是否正确")

    def __getitem__(self, index):
        file_path = self.image_list[index]
        label = self.label_list[index]

        # Use PIL to read the image
        img = Image.open(file_path).convert('RGB')

        if self.trans is not None:
            '''
            这个地方我认为通常是不需要做这种多余操作的
            '''
            img = self.trans(img)*255


        return img, label

    def __len__(self):
        return len(self.image_list)


if TEST:
    dataser=dataset_from_image_folder('/home/iiap/桌面/资料/cifar-10/train')

    data=torch.utils.data.DataLoader(dataser,batch_size=100,shuffle=True,num_workers=16)

    for i,j in data:
        print(i.shape)
        print(j)


class noisy_image_generator(Dataset):

    def __init__(self,channle :  int ,  width : int ,height : int):

        self.channel = channle

        self.height  =  height

        self.width = width

    def __getitem__(self, item):

        img = torch.rand((self.channel,self.width,self.height)).float()*254

        return  img

    def __len__(self):
        return int(1e10)


class noisy_vector_generator(Dataset):

    def __init__(self,length : int ):

        self.length = length

    def __getitem__(self, item):

        img = torch.rand(self.length)

        return  img

    def __len__(self):
        return int(1e10)

class onehot_vector_generator(Dataset):

    def __init__(self,classnum):

        self.class_num= classnum

    def __getitem__(self, item):

        img=torch.nn.functional.one_hot(torch.tensor(random.randint(0,9)),self.class_num).float()

        return img

    def __len__(self):

        return int(1e10)



class get_translation_from_raw_text_en2zh(Dataset):

    def __init__(self,en_text_path:str,zh_text_path:str):
        '''
        :discription: 两边都是一行对应一句话
        :return: batchsize,
        '''

        print("正在尝试初始化文本数据集")

        self.text_en = open(en_text_path)
        self.text_zh = open(zh_text_path)


        self.en_list  = []
        self.zh_list = []

        '''
        这种做法在遇到空文件的时候可能会出现错误
        '''

        temp="this is a start"

        while temp!="":

            self.en_list.append(self.text_en.tell())

            temp = self.text_en.readline()

        temp = "this is a start"

        while temp!="":

            self.zh_list.append(self.text_zh.tell())

            temp = self.text_zh.readline()

        print("初始化完毕")

    def __getitem__(self, item):

        self.text_zh.seek(self.zh_list[item])

        self.text_en.seek(self.en_list[item])

        return self.text_zh.readline().strip(),self.text_en.readline().strip()



    def __len__(self):

        return min(len(self.en_list),len(self.zh_list))




class get_translation_from__jsonlines_en2zh(Dataset):

    def __init__(self,en_zh_text_path:str):
        '''
        :discription: 两边都是一行对应一句话
        :return: batchsize,
        '''

        print("正在尝试初始化文本数据集")

        self.text_en_zh = open(en_zh_text_path)

        self.en_zh_list  = []


        '''
        这种做法在遇到空文件的时候可能会出现错误
        '''

        temp="this is a start"

        while temp!="":

            self.en_zh_list.append(self.text_en_zh.tell())

            temp = self.text_en_zh.readline()


        print("初始化完毕")

    def __getitem__(self, item):



        self.text_en_zh.seek(self.en_zh_list[item])

        source="hello"
        target="你好"

        temp = self.text_en_zh.readline().strip()

        try:
            source = json.loads(temp)["english"]
            target=json.loads(temp)["chinese"]

        except json.decoder.JSONDecodeError:
            print("load error!")
            print(temp)

        return source,target



    def __len__(self):

        return len(self.en_zh_list)



class get_translation_from__jsonlines_full(Dataset):

    def __init__(self,en_zh_text_path:str):
        '''
        :discription: 两边都是一行对应一句话
        :return: batchsize,
        '''

        print("正在尝试初始化文本数据集")

        self.en_list  = []
        self.zh_list = []

        '''
        这种做法在遇到空文件的时候可能会出现错误
        '''

        with jsonlines.open(en_zh_text_path) as file :

            for i  in file:

                self.en_list.append(i["english"])
                self.zh_list.append(i["chinese"])

        print("初始化完毕")

    def __getitem__(self, item):

        return self.en_list[item],self.zh_list[item]

    def __len__(self):

        return min(len(self.zh_list),len(self.en_list))

import pandas as pd

class read_gsmk8k(Dataset):
    """GSM8K Chinese数据集加载器"""

    def __init__(self, data_dir, sys_prompt,split="train"):
        """
        初始化数据集

        参数:
            data_dir (str): 包含parquet文件的目录
            split (str): 数据分割，"train"或"test"
        """
        # 构建parquet文件路径
        file_path = os.path.join(data_dir, f"{split}-00000-of-00001.parquet")

        # 读取parquet文件
        self.data = pd.read_parquet(file_path)

        self.SYSTEM_PROMPT=sys_prompt


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """获取单个数据样本"""
        item = self.data.iloc[idx]

        # 获取中文问题，如果不存在则使用英文问题
        question = item.get('question_zh-cn', item.get('question', ''))

        # 获取参考答案
        answer_only = item.get('answer_only', '')
        if not answer_only and 'answer' in item:
            answer_lines = item['answer'].strip().split('\n')
            if answer_lines:
                answer_only = answer_lines[-1]

        # 创建完整提示
        prompt = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        return {
            "prompt": prompt,
            "question": question,
            "answer": answer_only
        }




class read_jsonlines_dpo(Dataset):
    def __init__(self, json_path: str, tokenizer, max_length=2048):
        """
        初始化数据集
        :param json_path: JSONL文件路径
        :param tokenizer: 你的TokenizerUtil实例
        :param max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 读取所有数据行
        with open(json_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

    def __getitem__(self, item):
        """处理单条数据，返回模型需要的格式"""
        data = self.data[item]

        # 只处理chosen数据（根据你的data_generator逻辑）
        chosen = data['chosen']

        # 构建完整对话文本
        prompt = f"{chosen[0]['role']}:{chosen[0]['content']}"
        response = f"{chosen[1]['role']}:{chosen[1]['content']}"
        full_text = f"{prompt}\n{response}"

        # 编码完整文本
        input_ids, attention_mask = self.tokenizer.encode(full_text)

        # 计算response起始位置
        prompt_only = self.tokenizer.easy_encode(prompt)
        response_start = len(prompt_only)  # 获取prompt编码长度

        # 创建labels（将prompt部分设为-100）
        labels = input_ids.clone()
        labels[:response_start + 1] = -100  # +1 包含换行符

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)


class CocoDataset(Dataset):
    def __init__(self, root, annotation, transform=None, target_transform=None, size=(224, 224)):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = self.coco.getImgIds()
        self.transform = transform
        self.target_transform = target_transform
        self.size = size

    def __getitem__(self, index):
        coco = self.coco

        img_id = self.ids[index]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]

        path = img_info['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        # 图像尺寸归一化
        original_size = (img_info['width'], img_info['height'])

        # 如果有转换函数，应用它们

        # 调整标签
        targets = self.resize_annotations(annotations, original_size, self.size)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return image, targets

    def __len__(self):
        return len(self.ids)

    def resize_annotations(self, annotations, original_size, new_size):
        """
        调整标注的边界框尺寸。

        :param annotations: COCO 格式的标注数据
        :param original_size: 原始图像尺寸 (width, height)
        :param new_size: 新图像尺寸 (width, height)
        :return: 调整尺寸后的标注数据
        """
        orig_width, orig_height = original_size
        new_width, new_height = new_size
        x_scale = new_width / orig_width
        y_scale = new_height / orig_height

        new_annotations = []
        for ann in annotations:
            # COCO 的边界框格式是 [x_min, y_min, width, height]
            x, y, w, h = ann['bbox']

            # 调整边界框尺寸
            x = x /orig_width
            y = y /orig_height
            w = w /orig_width
            h = h /orig_height

            new_ann = ann.copy()
            new_ann['bbox'] = [x, y, x+w, y+h]
            new_annotations.append(new_ann)

        return new_annotations

# datasett= noisy_image_generator(224,224,3)
#
# for i in datasett:
#     print(i.shape)
#
#     cv2.imshow("test",i.byte().numpy())
#
#     cv2.waitKey(0)