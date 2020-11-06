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


# def aHash(img):
#     # 均值哈希算法
#     # 缩放为8*8
#     # 转换为灰度图
#     img = cv2.resize(img, (pic_length, pic_length))
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # s为像素和初值为0，hash_str为hash值初值为''
#     s = 0
#     hash_str = ''
#     # 遍历累加求像素和
#     for i in range(pic_length):
#         for j in range(pic_length):
#             s = s + gray[i, j]
#     # 求平均灰度
#     avg = s / (pic_length * pic_length)
#     # 灰度大于平均值为1相反为0生成图片的hash值
#     for i in range(pic_length):
#         for j in range(pic_length):
#             if gray[i, j] > avg:
#                 hash_str = hash_str + '1'
#             else:
#                 hash_str = hash_str + '0'
#     return hash_str
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
    # Hash值对比
    # 算法中1和0顺序组合起来的即是图片的指纹hash。顺序不固定，但是比较的时候必须是相同的顺序。
    # 对比两幅图的指纹，计算汉明距离，即两个64位的hash值有多少是不一样的，不同的位数越小，图片越相似
    # 汉明距离：一组二进制数据变成另一组数据所需要的步骤，可以衡量两图的差异，汉明距离越小，则相似度越高。汉明距离为0，即两张图片完全一样
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        n += abs(int(hash1[i]) - int(hash2[i]))
    return n


label_list = []
file = open('avg_output_256_label1.txt')

for i in file:
    photo = i.split('\t')[0]
    mid = i.split('\t')[1]
    tupe = (photo, mid)
    label_list.append(tupe)
# file.close()
file2 = open('avg_output_256_label0.txt')
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
for i in range(1):
    random = np.random.randint(0, len(path_list) - 1)
    ans = 99999999999999999
    count = 0
    # path = path_list[random]
    path = '/disk/data/total_incident/0/CME20011227060606/20011227_0830_c2_1024.jpg'
    print('#'+str(path))
    img = cv2.imread(path)
    hash = aHash(img)
    for i in label_list:
        count += 1
        if count % 1000 == 0:
            print('#has finish ', count)

        # if i[0] == path.split('/')[-1]:
        #     continue
        if i[0].split('/')[5] == path.split('/')[5]:
            continue
        # curhash=cmpHash(hash, i[1][0:65536])
        curhash = cmpHash(hash, i[1][0:1016])
        if ans > curhash:
            result_photo_name = i[0]
            ans = curhash
    print(str(path) +';' + str(result_photo_name) + '\n')
    print('#analyse finish ,picture is ', result_photo_name)
