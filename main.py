
import cv2
import os

def getImagePath(path):
    pathList = []
    for files in os.listdir(path):
        if files.endswith(('.jpg', '.jpeg', '.png', '.bmp')) and 'GT' not in files:
            image_path = os.path.join(path, files)
            pathList.append(image_path)
    return pathList



# Get image paths
paths = getImagePath("ball_frames/")
removed_images = paths[:17]
remaining_images = paths[17:]
outlist = remaining_images + removed_images

# Iterate over images
previous_image = cv2.imread(outlist[0], cv2.IMREAD_GRAYSCALE)
for idx, img_path in enumerate(outlist[1:]):
    current_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    result = cv2.subtract(current_image, previous_image)
    
    # Display the difference
    cv2.imshow('result', result)
    cv2.waitKey(0)
    
    # Update previous image
    previous_image = current_image

cv2.destroyAllWindows()

