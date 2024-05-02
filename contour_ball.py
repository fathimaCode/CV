import cv2
import numpy as np
# Load the image
input_image = cv2.imread('frame-54.png')
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
# Define the range of colors for the balls in HSV
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

image_hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

lower_white = (0, 0, 150)
upper_white = (255, 255, 255)
lower_orange = np.array([5, 100, 100])
upper_orange = np.array([15, 255, 255])
lower_yellow = (5, 100, 100)
upper_yellow = (15, 255, 255)

# Create masks for each color
mask_white = cv2.inRange(image_hsv, lower_white, upper_white)
mask_orange = cv2.inRange(image_hsv, lower_orange, upper_orange)
mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)

# Bitwise-OR the masks to combine them
mask = cv2.bitwise_or(mask_white, cv2.bitwise_or(mask_orange, mask_yellow))

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours and draw a circle around each one
for contour in contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    if radius > 10 and radius<25:
        cv2.circle(masked_image, center, radius, (0, 255, 0), 2)

# Save the output image
cv2.imwrite('output.jpg', masked_image)

