import cv2
import os
path  = '/disk/11711603/LAB/result_80.txt'
file = open(path)

for i in file:
    filename = i.split(';')
    testname = filename[0]
    findname = filename[1]
    photopath = os.path.dirname(testname)
    photopath = os.path.join(photopath,str(findname) )
    img = cv2.imread(testname)
    img2 = cv2.imread(photopath)
    while True:
        cv2.imshow('test.jpg', img)
    cv2.imshow('test.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.imshow('ans.jpg', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)

