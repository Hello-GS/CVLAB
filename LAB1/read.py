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
output_path = './output256.txt'
user, password, receiver, host = '', '', [], ''
year_map_0 = {}
year_map_1 = {}
year_event_0 = {}
year_event_1 = {}
event_graph_0 = {}
event_graph_1 = {}

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
        return 88888888888888888
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

path = '/disk/data/total_incident/1/CME19961219163007/19961219_1630_c2_1024.jpg'
path2 = '/disk/data/total_incident/0/CME19960814193009/19960814_1930_c2_1024.jpg'
path3 = '/disk/data/total_incident/0/CME19960814193009/19960814_2148_c2_1024.jpg'
img = cv2.imread(path)
img2 = cv2.imread(path2)
img3 = cv2.imread(path3)
file = open('/disk/11711603/LAB/LAB1/output2.txt')
sum=0
list = []
for i in file:
    # print(sum)
    if sum==44027:
        continue
    photo=i.split('\t')[0]
    mid=i.split('\t')[1]
    print(len(mid))
    if len(mid)!=257:
        continue
    # print(photo)
    # print(mid)
    sum+=1
    tupe = (photo,mid)
    list.append(tupe)

hash = aHash(img)

print(cmpHash(aHash(img),aHash(img3)))
str =''
ans = 99999999999999999
for i in list:
    # print( cmpHash(hash, i[1]))
    if i[0]=='19960814_1930_c2_1024.jpg':
        continue
    if ans > cmpHash(hash, i[1][0:256]):
        str = i[0]
        ans = cmpHash(hash, i[1][0:256])

print(ans)
print(str)

