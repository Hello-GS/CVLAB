import configparser
import cv2
import numpy as np
import yagmail
import os
import sys
import pandas as pd
import numpy as np
from fenjiaodu import angleHashCalculator

input_path = '/disk/data/total_incident/'
output_path_fre = ''
output_path_post = ''
user, password, receiver, host = '', '', [], ''
result_photo_name = ''
label_list_256 = []
label_list_32 = []
path_list = []
result_path = ''


class PriorityQueue:

    def __init__(self):
        self._queue = []
        self._min = 99999999999999999

    def push(self, item, priority):
        if self._min > priority:
            self._queue.clear()
            self._queue.append(item)
            self._min = priority
        if self._min == priority:
            self._queue.append(item)

    def get_list(self):
        return self._queue


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


def cmpHash(hash1, hash2, level):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    if level == 99999999999999999:
        for index in range(len(hash1)):
            n += abs(int(hash1[index]) - int(hash2[index]))
        return n
    for index in range(len(hash1)):
        n += abs(int(hash1[index]) - int(hash2[index]))
        if n > level:
            break
    return n


def read_config():
    config = configparser.ConfigParser()
    config.read('./config.ini')
    global user, password, receiver, input_path, output_path_fre, output_path_post, host, result_path
    input_path = config.get('data', 'data_path')
    output_path_fre = config.get('data', 'output_path_fre')
    output_path_post = config.get('data', 'output_path_post')
    user = config.get('email', 'user')
    password = config.get('email', 'password')
    receiver = config.get('email', 'receiver').split(',')
    host = config.get('email', 'host')
    result_path = config.get('data', 'result_path')


def read_feature(pic_size, label):
    file = open(output_path_fre + str(pic_size) + output_path_post + str(label) + '.txt')
    for line in file:
        cur = line.split('\t')
        if pic_size == 512:
            label_list_256.append((cur[0], cur[1]))
        if pic_size == 256:
            label_list_256.append((cur[0], cur[1]))
        else:
            label_list_32.append((cur[0], cur[1]))
    file.close()


def read_path():
    path_file = open('/disk/11712501/CVLAB/DHash/output_path.txt')
    for line in path_file:
        path_list.append(line[0:-1])


def send_email(title, content):
    yagmail.SMTP(user=user, password=password, host=host).send(receiver, title, content)


def match(label_list):
    q = PriorityQueue()
    ans = 99999999999999999
    for label in label_list:
        if label[0].split('/')[5] == path.split('/')[5]:
            continue
        cur_hash = cmpHash(target_hash, label[1][0:-1], ans)
        q.push(label[0], cur_hash)
    return q.get_list()


def calculate_answer(size, pre_ans):
    answer = 99999999999999999
    answer_path = ''
    if len(pre_ans) == 1:
        return pre_ans[0]
    for candidate in pre_ans:
        cur_pic = cv2.imread(candidate)
        cur_hash = hashCalculator.aHash(test_pic=cur_pic, pic_size=size)
        cur_ans = cmpHash(target_hash, cur_hash, answer)
        if answer > cur_ans:
            answer_path = candidate
            answer = cur_ans
    return answer_path

def sort_image_in_event(event_path):
    sort_image_list = []
    for image in os.listdir(event_path):
        sort_image_list.append(event_path+'/'+image)

    return sorted(sort_image_list)

def pair_similarity(df, image_list1, image_list2):

    simage = df['image_path']

    if len(image_list2) < len(image_list1):
        temp = image_list1
        image_list1 = image_list2
        image_list2 = temp

    # 默认行<列
    image_similarity_map = np.zeros([len(image_list1), len(image_list2)], dtype=int)
    i=-1
    #生成事件内图片map表
    for image1 in image_list1:
        i+=1
        j = -1
        #找到hash
        index1 = simage[simage.values == image1].index[0]
        hash1 = df.loc[index1,'hash']
        for image2 in image_list2:
            j+=1
            index2 = simage[simage.values == image2].index[0]
            hash2  = df.loc[index2,'hash']
            image_similarity_map[i][j] =cmpHash(hash1, hash2, 99999999999999999999)
    # 找到pair
    '''
        b1 b2 b3 b4 b5
    a1  0   0   0   0
    a2  0   0   0   0
    a3  0   0   0   0
    '''
    row_size = len(image_list1)

    start = 0
    ans_list = []  # 存储对应的B的索引 多余的A仍掉
    for i in range(row_size):
        #第一张图片的最近对
        if i==0:
            index_image_B = np.argmin(image_similarity_map[i][start:])
            start +=index_image_B
        else:
            start+=1
            index_image_B = np.argmin(image_similarity_map[i][start:])
            start += index_image_B

        ans_list.append(start)
        if (start+1) >= len(image_list2):
            break


    find_count = len(ans_list)
    assert find_count <= row_size

    ans_similarity = 0
    for i in range(find_count):
        ans_similarity += image_similarity_map[i][ans_list[i]]

    return ans_similarity/find_count





if __name__ == '__main__':
    f = open('just_for_test.txt', 'w')

    hashCalculator = HashCalculator()
    caculater512 = angleHashCalculator(512)
    read_config()
    # read_feature(pic_size=256, label=0)
    # read_feature(pic_size=256, label=1)
    # read_feature(pic_size=32, label=0)
    # read_feature(pic_size=32, label=1)
    read_feature(pic_size=512, label=0)
    read_feature(pic_size=512, label=1)
    print('read feature finish', flush=True)
    read_path()
    print('read path finish,length =', len(path_list), flush=True)


    #
    # 所有的事件放在一个list中
    item_list = []
    print("13232")
    for item in label_list_256:
        item_name = item[0].split('/')[5]
        if item_list.__contains__(item_name) is False:
            item_list.append(item_name)
    print("has put all items in one list")
    item_similarity_map = np.zeros([len(item_list), len(item_list)])

    print("the length of the items: ", len(item_list))

    image_list = []
    hash_list = []
    for item in label_list_256:
        image_list.append(item[0])
        hash_list.append(item[1][:-1])
    df = pd.DataFrame({'image_path': image_list,
                       'hash': hash_list
                        })

    # print(s)
    # print(s[s.values==df.loc[0,'image_path' ]].index)
    # print(df.loc[0,'image_path' ])
    # print(df.loc[29, 'image_path'])
    # print(df.loc[2950, 'image_path'])
    path0 = '/disk/data/total_incident/0'
    path1 = '/disk/data/total_incident/1'
    count = 1
    # i in 0
    for event_namei in os.listdir(path0):
        event_pathi = path0+'/'+event_namei

        #将一个事件内图片排序放入
        image_listi = sort_image_in_event(event_pathi)
        sizei = len(image_listi)

        #j in 0
        for event_namej in os.listdir(path0):

            event_pathj = path0+'/'+event_namej
            image_listj = sort_image_in_event(event_pathj)
            sizej = len(image_listj)

            #judge 3 days
            if event_namei[9] == '0':
                event_namei_day = int(event_namei[10])
            else:
                event_namei_day = int(event_namei[9:11])

            if event_namej[9] == '0':
                event_namej_day = int(event_namej[10])
            else:
                event_namej_day = int(event_namej[9:11])

            if abs(event_namei_day - event_namej_day) < 3:
                item_similarity_map[item_list.index(event_namei)][item_list.index(event_namej)] = sys.maxsize
                item_similarity_map[item_list.index(event_namej)][item_list.index(event_namei)] = sys.maxsize
                continue

            #pair 实现过程
            if item_similarity_map[item_list.index(event_namei)][item_list.index(event_namej)]!=0:
                continue
            similarity = pair_similarity(df, image_listi, image_listj)
            #对称
            item_similarity_map[item_list.index(event_namei)][item_list.index(event_namej)] = similarity
            item_similarity_map[item_list.index(event_namej)][item_list.index(event_namei)] = similarity



        #j in 1
        for event_namej in os.listdir(path1):

            event_pathj = path1 + '/' + event_namej
            image_listj = sort_image_in_event(event_pathj)
            sizej = len(image_listj)

            # judge 3 days
            if event_namei[9] == '0':
                event_namei_day = int(event_namei[10])
            else:
                event_namei_day = int(event_namei[9:11])

            if event_namej[9] == '0':
                event_namej_day = int(event_namej[10])
            else:
                event_namej_day = int(event_namej[9:11])

            if abs(event_namei_day - event_namej_day) < 3:
                item_similarity_map[item_list.index(event_namei)][item_list.index(event_namej)] = sys.maxsize
                item_similarity_map[item_list.index(event_namej)][item_list.index(event_namei)] = sys.maxsize
                continue

            # pair 实现过程
            if item_similarity_map[item_list.index(event_namei)][item_list.index(event_namej)] != 0:
                continue
            similarity = pair_similarity(df, image_listi, image_listj)
            # 对称
            item_similarity_map[item_list.index(event_namei)][item_list.index(event_namej)] = similarity
            item_similarity_map[item_list.index(event_namej)][item_list.index(event_namei)] = similarity

        find_list = item_similarity_map[item_list.index(event_namei)]
        index = np.argmin(find_list[:])
        result = event_namei + ":" + item_list[int(index)]
        print(result, flush=True)
        f.write(result+'\n')
        print('********')
        print(count)
        print('******')
        count+=1





    # i in 1
    for event_namei in os.listdir(path1):
        event_pathi = path1 + '/' + event_namei

        # 将一个事件内图片排序放入
        image_listi = sort_image_in_event(event_pathi)
        sizei = len(image_listi)

        # j in 0
        for event_namej in os.listdir(path0):
            event_pathj = path0 + '/' + event_namej
            image_listj = sort_image_in_event(event_pathj)
            sizej = len(image_listj)

            # judge 3 days
            if event_namei[9] == '0':
                event_namei_day = int(event_namei[10])
            else:
                event_namei_day = int(event_namei[9:11])
            if event_namej[9] == '0':
                event_namej_day = int(event_namej[10])
            else:
                event_namej_day = int(event_namej[9:11])

            if abs(event_namei_day - event_namej_day) < 3:
                item_similarity_map[item_list.index(event_namei)][item_list.index(event_namej)] = sys.maxsize
                item_similarity_map[item_list.index(event_namej)][item_list.index(event_namei)] = sys.maxsize
                continue

            # pair 实现过程
            if item_similarity_map[item_list.index(event_namei)][item_list.index(event_namej)] != 0:
                continue
            similarity = pair_similarity(df, image_listi, image_listj)
            # 对称
            item_similarity_map[item_list.index(event_namei)][item_list.index(event_namej)] = similarity
            item_similarity_map[item_list.index(event_namej)][item_list.index(event_namei)] = similarity

        # j in 1
        for event_namej in os.listdir(path1):
            event_pathj = path1 + '/' + event_namej
            image_listj = sort_image_in_event(event_pathj)
            sizej = len(image_listj)

            # judge 3 days
            if event_namei[9] == '0':
                event_namei_day = int(event_namei[10])
            else:
                event_namei_day = int(event_namei[9:11])

            if event_namej[9] == '0':
                event_namej_day = int(event_namej[10])
            else:
                event_namej_day = int(event_namej[9:11])

            if abs(event_namei_day - event_namej_day) < 3:
                item_similarity_map[item_list.index(event_namei)][item_list.index(event_namej)] = sys.maxsize
                item_similarity_map[item_list.index(event_namej)][item_list.index(event_namei)] = sys.maxsize
                continue

            # pair 实现过程
            if item_similarity_map[item_list.index(event_namei)][item_list.index(event_namej)] != 0:
                continue
            similarity = pair_similarity(df, image_listi, image_listj)
            # 对称
            item_similarity_map[item_list.index(event_namei)][item_list.index(event_namej)] = similarity
            item_similarity_map[item_list.index(event_namej)][item_list.index(event_namei)] = similarity

        find_list = item_similarity_map[item_list.index(event_namei)]
        index = np.argmin(find_list[:])
        result = event_namei + ":" + item_list[int(index)]
        f.write(result + '\n')
        print(count)
        count+=1

    #将表中最近事件挑选
    ans_event = []
    for i in range(len(item_list)):
        index = np.argmin(item_similarity_map[i][:])
        result = item_list[i] +":"+ item_list[int(index)]
        ans_event.append(result)
    print('event size: ', len(ans_event))
    with open('ans_pair_all.txt', 'w') as file:
        for line in ans_event:
            file.write(line+"\n")











