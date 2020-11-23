import  os
import cv2

save_path = '/disk/data/video/0/1112'+ '.mp4'
fps = 2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
size= (256,256)
video = cv2.VideoWriter(save_path, fourcc, fps, size)
for photo in os.listdir('/disk/data/total_incident/0/CME19960822083843'):
    photo_path = '/disk/data/total_incident/0/CME19960822083843/'+photo



    frame = cv2.imread(photo_path)
    image = cv2.resize(frame,(256,256))
    video.write(image)
video.release()