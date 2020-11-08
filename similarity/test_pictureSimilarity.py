def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        n += abs(int(hash1[i]) - int(hash2[i]))
    return n


import cv2
import math


def calculate_angle(i, j):
    x = i - 127
    y = j - 127
    return math.acos(x / math.sqrt(x * x + y * y))


def test_cmp_hash():
    valid = set()
    n = 0
    for i in range(256):
        for j in range(256):
            if (128 - i) * (128 - i) + (128 - j) * (128 - j) <= 128 * 128:
                valid.add((i, j))
    for t in range(100):
        for i in range(256):
            for j in range(256):
                if (i, j) in valid:
                    n += 1
    print(n)


def test_cmp():
    n = 0
    for t in range(100):
        for i in range(256):
            for j in range(256):
                if (128 - i) * (128 - i) + (128 - j) * (128 - j) <= 128 * 128:
                    n += 1
    print(n)
