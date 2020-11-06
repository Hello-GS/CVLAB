import configparser
import os
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt

input_path = '/disk/data/total_incident/'
output_path_label0 = './avg_output_256_label0.txt'
output_path_label1 = './avg_output_256_label1.txt'
user, password, receiver, host = '', '', [], ''
pic_length = 256


def read_config():
    config = configparser.ConfigParser()
    config.read('./config.ini')
    global user, password, receiver, input_path, output_path_label0,output_path_label1, host
    input_path = config.get('data', 'data_path')
    output_path_label0 = config.get('data', 'output_path_label0')
    output_path_label1 = config.get('data', 'output_path_label1')
    user = config.get('email', 'user')
    password = config.get('email', 'password')
    receiver = config.get('email', 'receiver').split(',')
    host = config.get('email', 'host')




def aHash(img):
    # 均值哈希算法
    # 缩放为8*8

    img = cv2.resize(img, (pic_length, pic_length))
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''

    new_gray = np.zeros([127,8])

    for i in range(1, 128):
        x = 127+i
        #new_gray[i-1][0] = gray[127][x]
        y = 127
        #右上
        count1=0
        value1=0
        while (127-y)!=i:
            count1+=1
            y = y-1
            value1+=gray[y][x]
        new_gray[i-1][1] = value1/count1

        #上
        count2=0
        value2=0
        while x!=127:
            count2+=1
            x=x-1
            value2+=gray[y][x]
        new_gray[i-1][2] = value2/count2

        #左上
        count3=0
        value3=0
        while (127-x)!=i:
            count3+=1
            x=x-1
            value3+=gray[y][x]
        new_gray[i-1][3] = value3/count3

        #左
        count4=0
        value4=0
        while (127-y)!=0:
            count4+=1
            y=y+1
            value4+=gray[y][x]
        new_gray[i-1][4] = value4/count4

        #左下
        count5=0
        value5=0
        while (y-127)!=i:
            count5+=1
            y=y+1
            value5+=gray[y][x]
        new_gray[i-1][5] = value5/count5

        #下
        count6=0
        value6=0
        while (127-x)!=0:
            count6+=1
            x=x+1
            value6+=gray[y][x]
        new_gray[i-1][6] = value6/count6

        #右下
        count7=0
        value7=0
        while (x-127)!=i:
            count7+=1
            x=x+1
            value7+=gray[y][x]
        new_gray[i-1][7] = value7/count7

        #回右
        count8=0
        value8=0
        while (y-127)!=0:
            count8+=1
            y=y-1
            value8+=gray[y][x]
        new_gray[i-1][0]=value8/count8

    sum=0
    count=0
    for i in range(127):
        for j in range(8):
            sum+=new_gray[i][j]
            count+=1
    avg = sum/count
    # 遍历累加求像素和
    for i in range(127):
        for j in range(8):
            if new_gray[i][j]>avg:
                hash_str += '1'
            else:
                hash_str+='0'
    return hash_str


def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        n += abs(int(hash1[i]) - int(hash2[i]))
    return n


def getImageByUrl(url):
    html = requests.get(url, verify=False)
    image = Image.open(BytesIO(html.content))
    return image


# 对图片进行统一化处理
def get_thum(image, size=(256, 256), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image


if __name__ == '__main__':
    read_config()
    # send_email('结果',output_path_label0)
    output = open(output_path_label0, 'r+')
    flies = os.listdir(input_path + '0/')
    count = 0
    for file in flies:
        if file == '.DS_Store':
            continue
        graphs = os.listdir(input_path + '0/' + file)
        for graph in graphs:
            curpath = '/disk/data/total_incident/' + '0/' + file + '/' + graph
            output.write(curpath + '\t' + aHash(cv2.imread(curpath)) + '\n')
            count += 1
            if count % 1000 == 0:
                print("has finish" + str(count))
    output = open(output_path_label1, 'r+')
    flies = os.listdir(input_path + '1/')
    for file in flies:
        if file == '.DS_Store':
            continue
        graphs = os.listdir(input_path + '1/' + file)
        for graph in graphs:
            curpath = '/disk/data/total_incident/' + '1/' + file + '/' + graph
            output.write(curpath + '\t' + aHash(cv2.imread(curpath)) + '\n')
            count += 1
            if count % 1000 == 0:
                print("has finish" + str(count))
