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
output_path = './output_256_label0.txt'
output_path2 = './output_256_label1.txt'
user, password, receiver, host = '', '', [], ''
pic_length = 256


def aHash(img):
    # 均值哈希算法
    # 缩放为8*8

    img = cv2.resize(img, (pic_length, pic_length))
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(pic_length):
        for j in range(pic_length):
            hash_str += str(int(gray[i, j] / 32))
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
    output = open(output_path, 'r+')
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
    output = open(output_path2, 'r+')
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

