import  os
import cv2


def task(video_read_path, video_save_path):
    for event in os.listdir(video_read_path):
        save_name = video_save_path + event+'.avi'
        video = cv2.VideoWriter(save_name, fourcc, fps, size)

        event_path = video_read_path + '/' + event

        for photo in os.listdir(event_path):
            photo_path = event_path + '/' + photo
            frame = cv2.imread(photo_path)
            frame = cv2.resize(frame,size)
            video.write(frame)
        video.release()
if __name__=='__main__':

    fps = 2
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
    size = (256, 256)

    video_save_path0 = '/disk/data/video/0/'
    video_save_path1 = '/disk/data/video/1/'

    video_read_path0 = '/disk/data/total_incident/0'
    video_read_path1 = '/disk/data/total_incident/1'
    #0
    task(video_read_path0, video_save_path0)
    task(video_read_path1, video_save_path1)

