import cv2
import os


video_file_path = './videos/video1.mp4'
save_dir = './frames/video1'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
    
vidcap = cv2.VideoCapture(video_file_path)
success, image = vidcap.read()
count = 0
results = []
while success:
    cv2.imwrite( save_dir + "/frame%04d.png" % count, image)     # save frame as PNG file      
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
