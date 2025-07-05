import os
import shutil
from PIL import Image

# 定义输入和输出文件夹路径
input_folders = ['/home/iiap/桌面/数据集/wds_vtab-cifar100/test/0', '/home/iiap/桌面/数据集/wds_vtab-cifar100/test/1', '/home/iiap/桌面/数据集/wds_vtab-cifar100/test/2',
                 '/home/iiap/桌面/数据集/wds_vtab-cifar100/test/3']
output_folder = '/home/iiap/PycharmProjects/再次开始的deeplearning/util/cifar100_test'

# 创建100个类别文件夹
for i in range(100):
    os.makedirs(os.path.join(output_folder, str(i)), exist_ok=True)

# 对每个输入文件夹进行处理
for folder in input_folders:
    for filename in os.listdir(folder):
        if filename.endswith('.webp'):
            # 获取图片文件名和类别文件名
            img_file = os.path.join(folder, filename)
            cls_file = os.path.join(folder, filename.replace('.webp', '.cls'))

            if os.path.isfile(cls_file):
                with open(cls_file, 'r') as f:
                    # 获取类别编号
                    cls = f.readline().strip()

                    # 确保类别编号是有效的
                    if cls.isdigit() and 0 <= int(cls) < 100:
                        output_path = os.path.join(output_folder, cls)

                        # 将图片复制到对应的类别文件夹里
                        shutil.copy(img_file, output_path)
                    else:
                        print(f"无效的类别编号: {cls} 文件: {cls_file}")
            else:
                print(f"未找到类别文件: {cls_file}")
        else:
            print(f"跳过非图片文件: {filename}")

print("处理完成！")