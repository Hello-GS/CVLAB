import  os
import cv2
import  numpy as np
save_path = '/disk/data/video/0/1112'+ '.mp4'
# fps = 2
# # # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# # # size= (256,256)
# # # video = cv2.VideoWriter(save_path, fourcc, fps, size)
# # # for photo in os.listdir('/disk/data/total_incident/0/CME19960822083843'):
# # #     photo_path = '/disk/data/total_incident/0/CME19960822083843/'+photo
# # #
# # #
# # #
# # #     frame = cv2.imread(photo_path)
# # #     image = cv2.resize(frame,(256,256))
# # #     video.write(image)
# # # video.release()

frame = cv2.imread('/disk/11711603/LAB/Video/2.png', cv2.IMREAD_GRAYSCALE)
print(frame.shape)
image_shape = frame.shape
imgs = np.zeros(shape=(image_shape[0], image_shape[1],3),dtype=np.float32)
print(imgs.shape)
imgs[:,:,0] = frame[:,:]
imgs[:,:,1] = frame[:,:]-100
imgs[:,:,2] = frame[:,:]-250
cv2.imwrite('/disk/11711603/LAB/Video/test8.png',imgs)