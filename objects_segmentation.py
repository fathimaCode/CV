import cv2
import os

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

# Get image paths
paths = getImagePath("ball_frames/")
gd_paths = getGnImagePath("ball_frames/")

for pa in gd_paths:
    print(pa)


yellowLower = (20, 100, 100)
yellowUpper = (40, 255, 255)
redLower = (0, 100, 100)
redUpper = (10, 255, 255)  