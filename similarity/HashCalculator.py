import cv2


class HashCalculator:
    valid_index = set()

    def __init__(self):
        for i in range(256):
            for j in range(256):
                if (128 - i) * (128 - i) + (128 - j) * (128 - j) <= 128 * 128:
                    self.valid_index.add((i, j))

    def aHash(self, img, pic_length):
        img = cv2.resize(img, (pic_length, pic_length))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hash_str = ''
        for i in range(pic_length):
            for j in range(pic_length):
                if (i, j) in self.valid_index:
                    hash_str += str(int(gray[i, j] / 32))
        return hash_str
