import cv2
import numpy as np
import glob
 
frames_path = './results/video1/*.png'
vidoe_file_name = 'video1.avi'
fps = 20
img_array = []
for filename in glob.glob(frames_path):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter(vidoe_file_name,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
    
out.release()
print("Video file save as ", vidoe_file_name)