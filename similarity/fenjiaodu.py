import math

import cv2

input_path = '/disk/data/total_incident/'
pic_length = 256


def calculate_angle(i, j):
    if i == 0 and j == 0:
        return 0
    print(i, j)
    x = i - 127
    y = j - 127
    if y < 0:
        return 360 - math.acos(x / math.sqrt(x * x + y * y)) / math.pi * 180
    elif y == 0:
        if x > +0:
            return 0
        else:
            return 180

    return math.acos(x / math.sqrt(x * x + y * y)) / math.pi * 180


def calculate_dest(i, j):
    return abs(i - 127) * abs(i - 127) + abs(j - 127) * abs(j - 127)


list = [[[] for j in range(5)] for i in range(36)]


def calculateAngle(fs):
    im = cv2.imread(fs, 0)
    im = cv2.resize(im, (pic_length, pic_length))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    for x in range(256):
        for y in range(256):
            if (128 - x) * (128 - x) + (128 - y) * (128 - y) <= 128 * 128:
                angle = int(calculate_angle(x, y) / 10)
                dist = calculate_dest(x, y)
                if dist < 1352:
                    list[angle][0].append(int(gray[x, y] / 8))
                elif dist < 5408:
                    list[angle][1].append(int(gray[x, y] / 8))
                elif dist < 12168:
                    list[angle][2].append(int(gray[x, y] / 8))
                elif dist < 21632:
                    list[angle][3].append(int(gray[x, y] / 8))
                else:
                    list[angle][4].append(int(gray[x, y] / 8))
    hash_number = []
    for i in range(36):
        for j in range(5):
            sum = 0
            for k in range(len(list[i][j])):
                sum += list[i][j][k]
            hash_number.append(int(sum / len(list[i][j])))
    print(hash_number)



if __name__ == '__main__':
    fs = '/disk/data/total_incident/1/CME20011104163506/20011104_1848_c2_1024.jpg'
    calculateAngle(fs)
