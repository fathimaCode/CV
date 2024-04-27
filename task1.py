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

def find_object_difference(prev_image, input_image):
    # Convert images to grayscale for simplicity
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between the two grayscale images
    diff_image = cv2.absdiff(prev_gray, input_gray)

    # Threshold the difference image to get binary image
    _, thresholded_diff = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded difference image
    contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the detected objects
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return input_image

paths = getImagePath("ball_frames/")
gd_paths = getGnImagePath("ball_frames/")
prev = None

for idx, img_path in enumerate(paths):
    input_image = cv2.imread(img_path)
    
    # Ensure we have a previous image to compare with
    if prev is not None:
        # Read the previous image
        prev_image = cv2.imread(paths[idx - 1])

        # Check if images have been successfully loaded
        if input_image is not None and prev_image is not None:
            # Find difference between objects in consecutive images
            result_image = find_object_difference(prev_image, input_image)
            
            # Display the result
            cv2.imshow("Result", result_image)
            cv2.waitKey(0)  # Press any key to proceed to the next image
            
            # Update prev_image for the next iteration
            prev = input_image
        else:
            print("Error: Couldn't load one of the images.")
    else:
        # First iteration, so just update prev_image
        prev = input_image

cv2.destroyAllWindows()