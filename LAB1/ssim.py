from skimage.measure import compare_ssim
import imageio
import numpy as np
import os

# filename1 = '19961219_1830_c2_1024.jpg'
#
# path1 = '/disk/data/total_incident/0/CME19961219183005/19961219_1830_c2_1024.jpg'
# path2 = '/Users/gaoshang/PycharmProjects/Comprehensive-design/19960814_2148_c2_1024.jpg'
#
# datapath = '/disk/data/total_incident'

ssim_socre = -1
classname = ''
list_ans = []
#
img = imageio.imread('/disk/data/total_incident/0/CME19960813071505/19960813_0715_c2_1024.jpg')
#
prename = ''
# print(img1.shape)

#ssim = compare_ssim(img1, img2, multichannel=True)





def task(dir):

    if os.path.isfile(dir):
        if os.path.basename(dir) != '.DS_Store':
            name = os.path.basename(dir)
            global prename
            if prename != name and name!='20120818_0636_c2_1024.jpg' and name!='20130715_1248_c2_1024.jpg' and name!='20130715_1548_c2_1024.jpg' and name!='20130715_1600_c2_1024.jpg' and name!='20120818_2312_c2_1024.jpg' and name!='20140212_0612_c2_1024.jpg':
                # select = random.randint(0,6)
                img2 = imageio.imread(dir)

                global img1
                img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))

                ssim = compare_ssim(img1, img2, multichannel=True)
                global ssim_socre
                global classname
                if ssim_socre < ssim:
                    ssim_socre = ssim
                    classname = dir

    elif os.path.isdir(dir):
        for i in os.listdir(dir):
            newdir = os.path.join(dir, i)
            task(newdir)


# task(datapath)
# print(classname)

def main():
    list_test = ['/disk/data/total_incident/0/CME19961219183005/19961219_1830_c2_1024.jpg',
                '/disk/data/total_incident/0/CME19960813071505/19960813_0715_c2_1024.jpg',
                 '/disk/data/total_incident/0/CME19970122133005/19970122_1330_c2_1024.jpg',
                 '/disk/data/total_incident/0/CME19970223023005/19970223_0550_c2_1024.jpg',
                 '/disk/data/total_incident/0/CME19980120220330/19980120_2359_c2_1024.jpg',
                 '/disk/data/total_incident/0/CME19980602025951/19980602_0533_c2_1024.jpg',
                 '/disk/data/total_incident/0/CME19980602025951/19980602_0631_c2_1024.jpg',
                 '/disk/data/total_incident/0/CME19980604020445/19980604_0400_c2_1024.jpg',
                 '/disk/data/total_incident/0/CME19980604212705/19980604_2200_c2_1024.jpg',
                 '/disk/data/total_incident/1/CME19970106151042/19970106_1734_c2_1024.jpg',
                 '/disk/data/total_incident/1/CME19970407142744/19970407_1521_c2_1024.jpg',
                 '/disk/data/total_incident/1/CME19971226023154/19971226_0417_c2_1024.jpg',
                 '/disk/data/total_incident/1/CME19971226023154/19971226_0706_c2_1024.jpg',
                 '/disk/data/total_incident/1/CME20000404163237/20000404_1806_c2_1024.jpg',
                 '/disk/data/total_incident/1/CME20000531080605/20000531_0906_c2_1024.jpg',
                 '/disk/data/total_incident/1/CME20000606155405/20000606_1730_c2_1024.jpg',
                 '/disk/data/total_incident/1/CME20000707102605/20000707_1126_c2_1024.jpg',
                 '/disk/data/total_incident/1/CME20011009113005/20011009_1331_c2_1024.jpg',
                 '/disk/data/total_incident/1/CME20020417082605/20020417_0906_c2_1024.jpg'
                 ]

    for path in list_test:
        print(path+"##############")
        global prename
        prename = path.split("/")[-1]
        global img1
        img1 = imageio.imread(path)
        global ssim_socre
        ssim_socre = -1
        global classname
        classname = ''
        task('/disk/data/total_incident')
        global list_ans
        print(classname)
        list_ans.append(classname)

main()
print(list_ans)


