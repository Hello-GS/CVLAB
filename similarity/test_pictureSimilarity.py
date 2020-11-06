import numpy as np
import cv2
pic_length =256
def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        n += abs(int(hash1[i]) - int(hash2[i]))
    return n
# def aHash(img):
#     # 均值哈希算法
#     # 缩放为8*8
#
#     img = cv2.resize(img, (pic_length, pic_length))
#     # 转换为灰度图
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # s为像素和初值为0，hash_str为hash值初值为''
#     s = 0
#     hash_str = ''
#     # 遍历累加求像素和
#     for i in range(pic_length):
#         for j in range(pic_length):
#             hash_str += str(int(gray[i, j] / 32))
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




path1 = '/disk/data/total_incident/0/CME20011226053005/20011226_1116_c2_1024.jpg'
img1 = cv2.imread(path1)
hash1 = aHash(img1)

path2 = '/disk/data/total_incident/0/CME20011227060606/20011227_0906_c2_1024.jpg'
img2 = cv2.imread(path2)
hash2= aHash(img2)
path3 ='/disk/data/total_incident/0/CME20011112092605/20011112_0926_c2_1024.jpg'
img3 = cv2.imread(path3)
hash3= aHash(img3)
def test_cmp_hash():
    print(cmpHash(hash1, hash2))
    print(cmpHash(hash1,hash3))
