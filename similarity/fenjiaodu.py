import math

import cv2



class angleHashCalculator():
    pic_length = 512
    list = [[[] for j in range(256)] for i in range(36)]


    def calculate_dest(self, i, j):
        return int(math.sqrt(
            abs(i - self.pic_length / 2) * abs(i - self.pic_length / 2) + abs(j - self.pic_length / 2) * abs(
                j - self.pic_length / 2)))

    def __init__(self, pic_length):
        self.pic_length = pic_length
        self.list = [[[] for j in range(int(pic_length / 2))] for i in range(36)]
        for i in range(pic_length):
            for j in range(pic_length):
                angel_level = int(self.calculate_angle(i, j) / 10)
                distance_level = int(self.calculate_dest(i, j))
                if distance_level >= pic_length / 2:
                    continue
                self.list[angel_level][distance_level].append((i, j))

    def calculateHash(self, fs):
        fs = cv2.imread(fs)
        img = cv2.resize(fs, (self.pic_length, self.pic_length))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ans_hash = ''
        for i in self.list:
            for j in i:
                if not j:
                    continue
                sum = 0
                for x, y in j:
                    sum += gray[x, y]
                hash_code = str(int(sum / len(j) / 32))
                ans_hash += hash_code
        return ans_hash


if __name__ == '__main__':
    fs = './20050108_1630_c2_1024.jpg'
    caculator = angleHashCalculator(512)
    print(caculator.calculateHash(fs))
