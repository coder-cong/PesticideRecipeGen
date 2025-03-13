
import json
import os
import os.path as op
import random
from time import sleep

import  cv2 as cv
# class_name_dict = {'blue_1': '0', 'blue_2': '1', 'blue_3': '2', 'blue_4': '3', 'blue_5': '4', 'blue_6': '5',
#                    'blue_7': '6', 'blue_8': '7', 'red_1': '8',  'red_2': '9',  'red_3': '10', 'red_4': '11',
#                    'red_5': '12', 'red_6': '13', 'red_7': '14', 'red_8': '15', 'robot': '16', 'outpost': '17',
#                    'Big_Luck': '18', 'R': '19', 'Bad_Luck': '20'
#                    }  # 步兵 英雄

def get_one_image(signal_image_path,signal_src_txt_path,roi_dir_path,out_img_width=32,out_img_height=32):
    '''
    :tip 具体哪些会保存哪些不会保存，写在这个函数里面
    :param signal_image_path: 单个图片地址
    :param signal_src_txt_path: 单个txt标签文件地址，使用yolo格式
    :param signal_roi_dir_path: 单个roi区域位置
    :param out_img_width: 输出图像宽度
    :param out_img_height: 输出图像高度
    :return:
    '''

    img = cv.imread(signal_image_path)
    # (1080, 1440, 3)
    print(img.shape)

    txt=open(signal_src_txt_path)
    temp=''
    counter=0

    while True:

        temp=txt.readline()
        if temp=='':
            break
        else:
            print(temp)
            values=temp.split(' ')

            category=int(values[0])
            if category==16:
                continue
            if category>7:
                category=category-8
            values=list(map(float,values[1:len(values)-1]))
            print(values[0])
            print(img.shape[1])
            point_x=int(img.shape[1]*values[0])
            point_y=int(img.shape[0]*values[1])
            part_wdth=int(0.5*img.shape[1]*values[2])
            part_height=int(0.5*img.shape[0]*values[3])

            print(point_y)
            print(point_x)
            print(part_height)
            print(part_wdth)

            img_part=img[point_y-part_height:(point_y+part_height),point_x-part_wdth:(point_x+part_wdth),:]

            print(img_part.shape)

            if(img_part.shape[0]== 0 or img_part.shape[1] == 0 ):
                continue
            img_part=cv.resize(img_part,(out_img_width,out_img_height),img_part)


            if os.path.isdir(os.path.join(roi_dir_path,str(category))):
                # 这个地方需要标注这个图片是从哪里来的

                cv.imwrite(os.path.join(roi_dir_path,str(category))+'/'+signal_image_path.split('/').pop()+'_'+str(random.randint(1,100000))+'.png',img_part)
            else:

                os.mkdir(os.path.join(roi_dir_path,str(category)))
                cv.imwrite(os.path.join(roi_dir_path,str(category))+'/'+signal_image_path.split('/').pop()+'_'+str(random.randint(1,100000))+'.png',img_part)









src_image_path = "/home/iiap/桌面/数据集/南区赛新增-car/"
# 这里采用标准的yolo格式进行装甲板roi区域的采集
src_txt_path="/home/iiap/桌面/数据集/南区赛txt/"

roi_dir_path='/home/iiap/PycharmProjects/再次开始的deeplearning/util/roi_images/roi/'

# get_one_image('/home/iiap/桌面/REPO/yolov5-face-master/data/data/images_car/171.png','/home/iiap/桌面/REPO/yolov5-face-master/data/data/labels_car/171.txt',
#               '/home/iiap/PycharmProjects/再次开始的deeplearning/util/roi_images')

if os.path.isdir(src_image_path) and os.path.isdir(src_txt_path) and os.path.isdir(roi_dir_path):
    image_list=os.listdir(src_image_path)
    txt_list=os.listdir(src_txt_path)

    def add_image_path(image_name):
        return os.path.join(src_image_path,image_name)
    def add_txt_path(txt_name):
        return os.path.join(src_txt_path,txt_name)

    for image_name in image_list :
        try:
            txt_list.index(image_name[:len(image_name)-4]+'.txt')

            get_one_image(add_image_path(image_name),add_txt_path(image_name[:len(image_name)-4]+'.txt'),roi_dir_path)

        except ValueError as v:
            continue
else:
    print("文件路径出错！！，请检查")




# txt_list=["1.txt",'2.txt',"3.txt"]
#
# image_name='1.png'
#
#
# print(txt_list.index(image_name[:len(image_name)-4]+'.txt'))


