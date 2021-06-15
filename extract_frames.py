import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='Args for file paths')
parser.add_argument('--file-path', default='./videos/video1.mp4',
                    help='video file path')
parser.add_argument('--save-dir', default='./frames/video1',
                    help='frames save dir')
args = parser.parse_args()

video_file_path = args.file_path
save_dir = args.save_dir
# print(video_file_path, save_dir)
# raise
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
