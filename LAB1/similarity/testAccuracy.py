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

pic_length = 256


def aHash(img):
    # 均值哈希算法
    # 缩放为8*8
    # 转换为灰度图
    img = cv2.resize(img, (pic_length, pic_length))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(pic_length):
        for j in range(pic_length):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / (pic_length * pic_length)
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(pic_length):
        for j in range(pic_length):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def cmpHash(hash1, hash2):
    # Hash值对比
    # 算法中1和0顺序组合起来的即是图片的指纹hash。顺序不固定，但是比较的时候必须是相同的顺序。
    # 对比两幅图的指纹，计算汉明距离，即两个64位的hash值有多少是不一样的，不同的位数越小，图片越相似
    # 汉明距离：一组二进制数据变成另一组数据所需要的步骤，可以衡量两图的差异，汉明距离越小，则相似度越高。汉明距离为0，即两张图片完全一样
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

label_list = []
file = open('output_256_label1.txt')

for i in file:
    photo = i.split('\t')[0]
    mid = i.split('\t')[1]
    tupe = (photo, mid)
    label_list.append(tupe)
# file.close()
file2 = open('output_256_label0.txt')
for i in file2:
    photo = i.split('\t')[0]
    mid = i.split('\t')[1]
    tupe = (photo, mid)
    label_list.append(tupe)
print('read file finish,length=', len(label_list))

result_photo_name = ''

path_file = open('../DHash/output_path.txt')
path_list = []

for line in path_file:
    path_list.append(line[0:-1])
print('#read path finish,length =', len(path_list))
output = open('./output_200_random_ans.txt', 'r+')
for i in range(200):
    random = np.random.randint(0, len(path_list) - 1)
    ans = 99999999999999999
    count = 0
    path = path_list[random]
    print('#'+str(path))
    img = cv2.imread(path)
    hash = aHash(img)
    for i in label_list:
        count += 1
        if count % 1000 == 0:
            print('#has finish ', count)
        if i[0] == path.split('/')[-1]:
            continue
        if ans > cmpHash(hash, i[1][0:65536]):
            result_photo_name = i[0]
            ans = cmpHash(hash, i[1][0:65536])
    print(str(path) +';' + str(result_photo_name) + '\n')
    print('#analyse finish ,picture is ', result_photo_name)
