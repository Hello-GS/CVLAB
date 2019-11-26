import cv2


def Hash(test_pic, pic_size, index_set):
    test_pic = cv2.resize(test_pic, (pic_size, pic_size))
    gray = cv2.cvtColor(test_pic, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for (i, j) in index_set:
        hash_str += str(int(gray[i, j] / 32))
    return hash_str


class HashCalculator:
    valid_index_256 = []
    valid_index_32 = []

    def __init__(self):
        for i in range(256):
            for j in range(256):
                if (128 - i) * (128 - i) + (128 - j) * (128 - j) <= 128 * 128:
                    self.valid_index_256.append((i, j))
        for i in range(32):
            for j in range(32):
                if (16 - i) * (16 - i) + (16 - j) * (16 - j) <= 16 * 16:
                    self.valid_index_32.append((i, j))

    def aHash(self, test_pic, pic_size):
        if pic_size == 32:
            return Hash(test_pic=test_pic, pic_size=pic_size, index_set=self.valid_index_32)
        elif pic_size == 256:
            return Hash(test_pic=test_pic, pic_size=pic_size, index_set=self.valid_index_256)
