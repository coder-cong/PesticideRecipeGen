import os

dir_name='/home/iiap/桌面/资料/Anotations/'

if os.path.isdir(dir_name):

    name_list=os.listdir(dir_name)

    for i in name_list:

        temp=i

        temp=temp.replace(':','')
        print(temp)
        os.rename(dir_name+i,dir_name+temp)

