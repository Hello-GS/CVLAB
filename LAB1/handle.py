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
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def getImageByUrl(url):
    # 根据图片url 获取图片对象
    html = requests.get(url, verify=False)
    image = Image.open(BytesIO(html.content))
    return image


def runAllImageSimilaryFun(para1, para2):
    # 均值、差值、感知哈希算法三种算法值越小，则越相似,相同图片值为0
    # 三直方图算法和单通道的直方图 0-1之间，值越大，越相似。 相同图片为1

    # t1,t2   14;19;10;  0.70;0.75
    # t1,t3   39 33 18   0.58 0.49
    # s1,s2  7 23 11     0.83 0.86  挺相似的图片
    # c1,c2  11 29 17    0.30 0.31

    if para1.startswith("http"):
        # 根据链接下载图片，并转换为opencv格式
        img1 = getImageByUrl(para1)
        img1 = cv2.cvtColor(np.asarray(img1), cv2.COLOR_RGB2BGR)

        img2 = getImageByUrl(para2)
        img2 = cv2.cvtColor(np.asarray(img2), cv2.COLOR_RGB2BGR)
    else:
        # 通过imread方法直接读取物理路径
        img1 = cv2.imread(para1)
        img2 = cv2.imread(para2)

    hash1 = aHash(img1)
    hash2 = aHash(img2)
    n1 = cmpHash(hash1, hash2)
    print('均值哈希算法相似度aHash：', n1)


# photoes = os.listdir('image/')
#
# for a in photoes:
#     for b in photoes:
#         if a != b:
#             print(a + "and " + b)
#             runAllImageSimilaryFun("image/" + a, "image/" + b)


def get_year(file_name):
    return file_name[3:7]


def write_map(map_to_write):
    out = ''
    for key in sorted(map_to_write.keys()):
        out += str(key) + ':' + str(map_to_write[key]) + '\n'
    return out


if __name__ == '__main__':
    output = open(output_path, 'r+')

    count = 0


    flies = os.listdir(input_path + '1/')
    for file in flies:
        if file == '.DS_Store':
            continue
        graphs = os.listdir(input_path + '1/' + file)
        for graph in graphs:
            curpath = '/disk/data/total_incident/' + '1/' + file + '/' + graph
            output.write(graph + '\t' + aHash(cv2.imread(curpath)) + '\n')
            count += 1
            if count % 1000 == 0:
                print("has finish" + str(count))