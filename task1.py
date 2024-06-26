import cv2
import os
import numpy as np
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

def segment_balls(input_image):
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Perform morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate = cv2.dilate(binary, kernel, iterations=2)
    erode = cv2.erode(dilate, kernel, iterations=2)

            # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(erode, connectivity=8)

            # Create a random color map
    colormap = np.zeros((num_labels, 3), dtype=np.uint8)
    colormap[1:, :] = np.random.randint(0, 255, size=(num_labels - 1, 3))

            # Apply the color map to the labels
    labeled_img = colormap[labels]

    
    # Convert input image to HSV color space
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for white, yellow, and orange colors in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])

    # Threshold the HSV image to obtain binary masks for each color
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

    # Combine masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    combined_mask = cv2.bitwise_or(combined_mask, orange_mask)

    # Perform morphological operations to refine the masks
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #cv2.imshow(" Detected Objects", combined_mask) 
    return combined_mask
    # Display the input image with bounding boxes
    #cv2.imshow("segmented Objects", input_image) 



def mask_image(input_image, start_point, end_point, rectangle_color=(255, 0, 0), mask_color=(0, 0, 0)):
    # Draw rectangle on the image
    image_with_rectangle = cv2.rectangle(input_image.copy(), start_point, end_point, rectangle_color, 3)

    # Create a mask with the same size as the input image
    mask = np.zeros_like(input_image)
    mask[:, :] = [255, 255, 255]  # Flip the mask color to white

    # Draw rectangle on the mask
    mask = cv2.rectangle(mask, start_point, end_point, mask_color, -1)

    # Apply the mask to the input image
    masked_image = cv2.bitwise_and(input_image, mask)

    return image_with_rectangle, masked_image
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




scores= []
paths = getImagePath("ball_frames/")
gd_paths = getGnImagePath("ball_frames/")

prev_image = None  # Initialize previous image variable

for idx, img_path in enumerate(paths):
    
    input_image = cv2.imread(img_path)
    #cv2.imshow(f"result {idx}", input_image)
    #cv2.imshow("Input Image", input_image)
    # masked image
    start_point = (0, 0)
    end_point = (1250, 330)
    sp1 = (0, 326)
    ep1 = (293, 385)
    sp2 = (379, 314)
    ep2 = (412, 352)
    sp3 = (1244, 52)
    ep3 = (1189, 297)
    sp4 = (0, 538)
    ep4 = (220,  719)
    sp5 = (1095, 0)
    ep5 = (1278,  316)
    image_with_rectangles, masked_image = mask_image(input_image, start_point, end_point, (255, 0, 0))
    _, masked_image = mask_image(masked_image, sp1, ep1)
    _, masked_image = mask_image(masked_image, sp2, ep2)
    _, masked_image = mask_image(masked_image, sp3, ep3)
    _, masked_image = mask_image(masked_image, sp4, ep4)
    _, masked_image = mask_image(masked_image, sp5, ep5)

    current_image = masked_image

    if prev_image is not None:
        diff = cv2.absdiff(prev_image, current_image)
        seg_balls = segment_balls(masked_image)
        cv2.imwrite(f"mask_image/{idx}.png", seg_balls)
        print(f"image idx:{idx}")
        ground_image = cv2.imread(gd_paths[idx], cv2.COLOR_BGR2GRAY)
        
        intersection = np.logical_and(seg_balls, ground_image)
        
        dice_score = 2.0 * np.sum(intersection) / (np.sum(seg_balls) + np.sum(ground_image))
        #print(dice_score)
        scores.append(dice_score)
        cv2.imshow(f"image idx", seg_balls)

    prev_image = current_image  
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cv2.destroyAllWindows()


mean_dice_score = np.mean(scores)
print(mean_dice_score)