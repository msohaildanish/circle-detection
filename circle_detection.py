import os
import glob
import numpy as np
import cv2
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from config import *
from tracker import CentroidTracker
from trackableobject import TrackableObject
import argparse

IMAGES_PATH =None
SAVE_DIR = None
# initialize our centroid tracker and frame dimensions
ct = CentroidTracker(maxDisappeared=MAX_DISAPPEARED, maxDistance=70)
(H, W) = (None, None)
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
x = []
empty=[]
empty1=[]

def sharpen(img):
    kernal_sharpening = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpend = cv2.filter2D(img,-1,kernal_sharpening)
    return sharpend

def nms(boxes, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
        np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def detect(file, acc_thresh=0.35, mean_thresh=160, totalDown=0):
    img = cv2.imread(file)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = img.copy()
    # image = sharpen(image)
    (H, W) = img.shape[:2]
    edges = canny(image, sigma=2, low_threshold=1, high_threshold=10)
    # Detect two radii
    hough_radii = np.arange(16, 30, 2)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii)
    # Draw them
    boxes = []
    means = []
    for acc, center_y, center_x, radius in zip(accums, cy, cx, radii):
        bx = [center_x-radius, center_y-radius, center_x+radius, center_y+radius ]
        box = image[center_y:center_y + radius, center_x:center_x+radius]
        mean_val = box.mean()
        means.append(mean_val)
        if acc > acc_thresh and mean_val > mean_thresh and center_y > 200:
            boxes.append(bx)

    bboxes = np.array(boxes)
    picks = nms(bboxes, overlapThresh=NMS_OVERLAP_THRESH)
    objects = ct.update(picks)
    # print(len(picks), len(bboxes))
    for (startX, startY, endX, endY) in picks:
        radius = (endX - startX)//2
        center_y = startY + radius
        center_x = startX + radius
        circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
        output[circy, circx] = CRICLE_COLOR
    
    # print the counter and the line
    # j =  len(picks)-1
    counter_center_H = (H // 3) * 2
    cv2.line(output, (0, counter_center_H), (W, counter_center_H), (0, 0, 0), 3)
    # cv2.putText(output, "-Counter border", (10, H // 2),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
       
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                if centroid[1] > counter_center_H:
                    totalDown += 1
                    empty1.append(totalDown)
                    to.counted = True
                    

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to
       
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(output, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # cv2.circle(output, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
        

    info2 = [
    ("Total", totalDown),
    ]

    for (i, (k, v)) in enumerate(info2):
        text = "{}: {}".format(k, v)
        cv2.putText(output, text, (2, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output, totalDown



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for file paths')
    parser.add_argument('--images-path', default='./frames/video1',
                        help='video file path')
    parser.add_argument('--save-dir', default='./results/video1',
                        help='frames save dir')
    args = parser.parse_args()

    IMAGES_PATH = args.images_path
    SAVE_DIR = args.save_dir
    # Create save dir if not exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # get paths of all images
    images = sorted(glob.glob(IMAGES_PATH + '/*'))
    print('Total images', len(images))
    for i, image in enumerate(images):
        # if i < 300:
        #     continue
        # detect and save the result image
        img, totalDown = detect(image, acc_thresh=ACC_THRESH, mean_thresh=MEAN_THRESH, totalDown=totalDown)
        cv2.imwrite(SAVE_DIR + '/' + image.split('/')[-1], img)     # save frame as JPEG file      
        print("Saved Frame# ", i+1)
        # if i == 300:
        #     break

