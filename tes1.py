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