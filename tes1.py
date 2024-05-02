import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
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
# Read image
img = img_list[2]



# Load the image
image = img
if image is None:
    print("Error: Unable to load image.")
    exit()

# Create a copy of the image for displaying the selections
clone = image.copy()

# Function to handle mouse events for selecting circles
circles = []

def select_circle(event, x, y, flags, param):
    global circles
    
    if event == cv2.EVENT_LBUTTONDOWN:
        circles.append((x, y))

# Display the image and select circles
cv2.namedWindow('Select Circles')
cv2.setMouseCallback('Select Circles', select_circle)

while True:
    cv2.imshow('Select Circles', clone)
    key = cv2.waitKey(1) & 0xFF
    
    # Press 'r' to reset selections
    if key == ord('r'):
        circles = []
        clone = image.copy()
    
    # Press 'c' to finish selecting circles
    elif key == ord('c'):
        break

# Create masks for each selected circle
masks = []
for center in circles:
    mask = np.zeros_like(image[:, :, 0])
    cv2.circle(mask, center, 50, (255, 255, 255), -1)  # Change radius as needed
    masks.append(mask)

# Display the masks
for idx, mask in enumerate(masks):
    cv2.imshow(f'Mask {idx+1}', mask)

cv2.waitKey(0)
cv2.destroyAllWindows()


""" blurred = cv2.GaussianBlur(previous_image, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
yellowLower = (20, 100, 100)
yellowUpper = (40, 255, 255)
redLower = (10, 100, 100)
redUpper = (25, 255, 255)  

yellow_mask = cv2.inRange(hsv, yellowLower, yellowUpper)
yellow_mask = cv2.erode(yellow_mask, None, iterations=2)
yellow_mask = cv2.dilate(yellow_mask, None, iterations=2)

red_mask = cv2.inRange(hsv, redLower, redUpper)
red_mask = cv2.erode(red_mask, None, iterations=2)
red_mask = cv2.dilate(red_mask, None, iterations=2)

combined_mask = cv2.bitwise_or(yellow_mask, red_mask)

result = cv2.bitwise_and(blurred, blurred, mask=combined_mask)


# Create a background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Apply background subtraction
foreground_mask = bg_subtractor.apply(previous_image)

# Optionally, you can perform morphological operations for noise removal
kernel = np.ones((5, 5), np.uint8)
foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel) """