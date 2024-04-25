import cv2
import os
import numpy as np

yellowLower = (20, 100, 100)
yellowUpper = (40, 255, 255)
redLower = (0, 100, 100)
redUpper = (10, 255, 255)  
whiteLower = (0, 0, 180)  
whiteUpper = (180, 25, 255)  

def readImages(path):
    image_list = []
    for files in os.listdir(path):
        if files.endswith(('.jpg', '.jpeg', '.png', '.bmp')) and 'GT' not in files:
            image_path = os.path.join(path, files)
            ball_image = cv2.imread(image_path)
            if ball_image is not None:
                image_list.append(ball_image)
    return image_list

img_list = readImages("ball_frames/")
blurred = cv2.GaussianBlur(img_list[5], (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

yellow_mask = cv2.inRange(hsv, yellowLower, yellowUpper)
yellow_mask = cv2.erode(yellow_mask, None, iterations=2)
yellow_mask = cv2.dilate(yellow_mask, None, iterations=2)

red_mask = cv2.inRange(hsv, redLower, redUpper)
red_mask = cv2.erode(red_mask, None, iterations=2)
red_mask = cv2.dilate(red_mask, None, iterations=2)

white_mask = cv2.inRange(hsv, whiteLower, whiteUpper)
white_mask = cv2.erode(white_mask, None, iterations=2)
white_mask = cv2.dilate(white_mask, None, iterations=2)

# Combine masks using bitwise OR operation
combined_mask = cv2.bitwise_or(cv2.bitwise_or(yellow_mask, red_mask), white_mask)


# Apply the combined mask to the original image
result = cv2.bitwise_and(blurred, blurred, mask=combined_mask)


# Convert the image to grayscale
gray_image = cv2.cvtColor(img_list[5], cv2.COLOR_BGR2GRAY)

# Create a background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Apply background subtraction
foreground_mask = bg_subtractor.apply(img_list[5])

# Optionally, you can perform morphological operations for noise removal
kernel = np.ones((5, 5), np.uint8)
foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)

cv2.imshow("Combined Mask", combined_mask)
cv2.imshow('Foreground Mask', foreground_mask)
cv2.imshow("Result", result)
cv2.imshow("Blurred Frame", blurred)
cv2.waitKey(0)

cv2.destroyAllWindows()
