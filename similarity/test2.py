import os


def read_path():
    path_list = []
    path_file = open('../DHash/output_path.txt')
    for line in path_file:
        path_list.append(line[0:-1])

    print(path_list)

path_list = []


def task(dir):
    if os.path.isfile(dir):

        if os.path.basename(dir) != '.DS_Store':
            #select = random.randint(0,6)
            global path_list
            path_list.append(str(dir))

        # print('wenjian'+dir)
        # print(os.path.basename(dir))
    elif os.path.isdir(dir):

        for i in os.listdir(dir):
            newdir = os.path.join(dir, i)
            task(newdir)
def write_class():
    global path_list
    with open('path_test1class.txt','a') as file:
        for i in path_list:
            file.write(i+'\n')




task('/disk/data/total_incident/1')
write_class()
