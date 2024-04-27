import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def getImagePath(path):
    pathList = []
    for files in os.listdir(path):
        if files.endswith(('.jpg', '.jpeg', '.png', '.bmp')) and 'GT' not in files:
            image_path = os.path.join(path, files)
            pathList.append(image_path)
    removed_images = pathList[:17]
    remaining_images = pathList[17:]
    pathList = remaining_images + removed_images
    return pathList

def getGnImagePath(path):
    pathList = []
    for files in os.listdir(path):
        if files.endswith('_GT.png'):
            image_path = os.path.join(path, files)
            pathList.append(image_path)
    removed_images = pathList[:17]
    remaining_images = pathList[17:]
    pathList = remaining_images + removed_images
    return pathList



paths = getImagePath("ball_frames/")
gd_paths = getGnImagePath("ball_frames/")
prev = 0

for idx, img in enumerate(paths):
    print(idx)
    previous_image =cv2.imread( paths[prev],0)
    
    if idx!=prev:
        current_image  =cv2.imread( paths[idx],0)
        diff = 255 - cv2.absdiff(previous_image, current_image)


        cv2.imshow("result: ",diff)
        prev = current_image

cv2.waitKey(0)

cv2.destroyAllWindows()
    