import json
import os
import os.path as op



'''
json.loads 用于解码 JSON 数据。该函数返回 Python 字段的数据类型。
'''

class_name_dict={'blue_1':'0', 'blue_2':'1', 'blue_3':'2', 'blue_4':'3', 'blue_5':'4', 'blue_6':'5', 'blue_7':'6', 'blue_8':'7', 'red_1':'8', 'red_2':'9', 'red_3':'10',
           'red_4':'11', 'red_5':'12', 'red_6':'13', 'red_7':'14', 'red_8':'15','robot':'16'}

# class_name_dict={'Big_Luck':'0','Bad_Luck':'2','R':'1'}

point_name_dict={''}


json_dir ="/media/iiap/新加卷/数据集/NEW/annotations/"

output_dir="/media/iiap/新加卷/数据集/NEW/txt/"



if op.isdir(json_dir):
    path_list=os.listdir(json_dir)

    if os.path.isdir(output_dir):
        print("exist")
    else:
        os.mkdir(output_dir)


    
    for i in path_list:

        '''
        json结构如下：
        shapes（数组）：
            label

            points(二维数组)

            shape_type：
            point
            ！！！；这里是按照两点法保存的，理论上我应该转化一下
            !!!:    前四个值我需要换成中心坐标加wh.
            rectangle

        'imageHeight': 1080
        'imageWidth': 1440
         '''
        json_text = open(json_dir+i)

        json_data = json.loads(json_text.read())

        rectangle_list=[]

        point_list=[]
        #这个里面就之保存字符串，不关心数据类型，怎么写入就怎么写出
        final_list=[]
        #提取出所有的形状

        width=float(json_data['imageWidth'])

        height=float(json_data['imageHeight'])

        #创建文件并且覆盖原来的文件

        print(i)
        txt=open(output_dir+i.replace('.json','')+'.txt','w')

        for shape in json_data['shapes']:

            if shape['shape_type']=='rectangle':

                rectangle_list.append(shape)

            elif shape['shape_type']=='point':

                point_list.append(shape)

        for rectange in rectangle_list:

            object=[]
            '''
            class_id x,y,w,h,x1,y1,x2,y2,x3,y3,x4,y4
            '''
            if rectange['label'] in class_name_dict:

                object.append(class_name_dict[rectange['label']])

                object.append(float(rectange['points'][0][0] + rectange['points'][1][0]) / (2 * width))  # x

                object.append(float(rectange['points'][0][1] + rectange['points'][1][1]) / (2 * height))  # y

                object.append(float(rectange['points'][1][0] - rectange['points'][0][0]) / width)  # imageWidth

                object.append(float(rectange['points'][1][1] - rectange['points'][0][1]) / height)  # height

                lu = []

                ld = []

                ru = []

                rd = []

                # 找距离中心点最近的左上角点
                for p in point_list:

                    # 判断是否在框内
                    if (p['points'][0][0] > rectange['points'][0][0]) & (
                            p['points'][0][0] < rectange['points'][1][0]) & (
                            p['points'][0][1] > rectange['points'][0][1]) & (
                            p['points'][0][1] < rectange['points'][1][1]):

                        # 最后还是选取离框中心点距离最近的
                        if p['label'] == 'left_up':
                            lu.append(p['points'])
                            # print("points")

                        elif p['label'] == 'left_down':
                            ld.append(p['points'])
                            # print('points')


                        elif p['label'] == 'right_up':
                            ru.append(p['points'])
                            # print('points')


                        elif p['label'] == 'right_down':
                            rd.append(p['points'])
                            # print('points')

                # 进行匹配

                min_distans = 1000000
                final_point = []

                for LU in lu:

                    for LD in ld:

                        for RU in ru:

                            for RD in rd:

                                # 取对角线交点较为复杂，时间问题采用平均值

                                x_mean = (LU[0][0] + LD[0][0] + RU[0][0] + RD[0][0]) / 4

                                y_mean = (LU[0][1] + LD[0][1] + RU[0][1] + RD[0][1]) / 4

                                distance = (abs(x_mean - float(
                                    rectange['points'][0][0] + rectange['points'][1][0]) / 2) + abs(
                                    y_mean - float(rectange['points'][0][1] + rectange['points'][1][1]) / (2)))

                                if distance < min_distans:
                                    min_distans = distance

                                    final_point.clear()

                                    final_point.append(LU[0][0] / width)
                                    final_point.append(LU[0][1] / height)
                                    final_point.append(LD[0][0] / width)
                                    final_point.append(LD[0][1] / height)
                                    final_point.append(RU[0][0] / width)
                                    final_point.append(RU[0][1] / height)
                                    final_point.append(RD[0][0] / width)
                                    final_point.append(RD[0][1] / height)

                # 封装成一个object,这是一个方框的，然后输出一下看看

                for f in final_point:
                    object.append(f)
                # 作长度的合格检查

                #对于机器人类别做修正：
                print(object[0])
                if (object[0]=='16'):

                    x=0.0
                    y=0.0

                    x=object[1]
                    y=object[2]

                    if(len(object)==5):

                        for k  in range(4):
                            object.append(x)
                            object.append(y)

                    if(len(object)==13):
                        for k in range(8):
                            object.pop()
                        for k in range(4):
                            object.append(x)
                            object.append(y)
                if (len(object) != 13):
                    print('length error please check the file !!')
                    print(object)
                    print(i)
                    exit(-1)

                # 尝试写如文件

                for i in object:
                    txt.write(str(i))
                    txt.write(' ')

                txt.write('\n')
                print('write to:' + txt.name)



        txt.close()








                                    























