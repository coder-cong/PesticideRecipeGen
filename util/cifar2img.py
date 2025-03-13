import os.path

import cv2
import numpy as np
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def show_img_from_array(array):
    img = np.array(array )
    img = img.reshape((3, 32, 32)).transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("test", img)
    cv2.waitKey(0)

def save_img_from_array(array,filename,class_id,file_path):

        dir_path=os.path.join(file_path,str(class_id))

        if os.path.isdir(dir_path) == False:

            os.mkdir(dir_path)


        dir_path=os.path.join(dir_path,filename.decode('utf-8'))

        img = np.array(array)

        img = img.reshape((3, 32, 32)).transpose(1, 2, 0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow("test", img)
        # cv2.waitKey(1)
        cv2.imwrite(dir_path,img)


    # cv2.destroyWindow("test")


img_dir="/home/iiap/桌面/数据集/cifar10/test"

dick = unpickle("/home/iiap/桌面/数据集/cifar-10-batches-py/test_batch")
'''
b'filenames'
b'batch_label'
b'fine_labels'
b'coarse_labels'
b'data'


b'batch_label'
b'labels'
b'data'
b'filenames'

'''
for k ,v in dick.items():
    print(k)



for img,class_id,filename  in zip(dick[b'data'],dick[b'labels'],dick[b'filenames']):

    print(class_id)
    print(filename)
    save_img_from_array(img, filename, class_id, file_path=img_dir)






