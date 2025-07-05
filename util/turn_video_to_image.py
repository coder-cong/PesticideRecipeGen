import cv2

import json

import time

cap=cv2.VideoCapture("file:///home/iiap/视频/2022-07-29 11-39-02.mp4")

save_dir="/home/iiap/桌面/数据集/南区赛新增-car/"

def get_current_time():
    """
    [summary] 获取当前时间
    [description] 用time.localtime()+time.strftime()实现
    :returns: [description] 返回str类型
    """
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d-%H-%M-%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp


while cap.isOpened():

    istrue,frame = cap.read()

    # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    cv2.imshow("test",frame)


    key=cv2.waitKey(0)

    print(key)
    #s
    if key==115:
        print(save_dir+get_current_time()+".png")
        cv2.imwrite(save_dir+get_current_time()+".png",frame)
    #d
    elif key==100:

        continue