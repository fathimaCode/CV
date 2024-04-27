#check the mask
import cv2
import numpy as np

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

# Example usage
if __name__ == "__main__":
    paths = ["frame-54.png"]  # Replace with your image path
    input_image = cv2.imread(paths[0])

    start_point = (0, 0)
    end_point = (1250, 330)

    sp1 = (0, 326)
    ep1 = (293, 385)

    sp2 = (380, 314)
    ep2 = (429, 347)

    sp3 = (1244, 52)
    ep3 = (1189, 297)


    sp4 = (0, 538)
    ep4 = (220,  719)
    sp5 = (1095, 0)
    ep5 = (1278,  316)




    # Mask the image with two rectangles
    image_with_rectangles, masked_image = mask_image(input_image, start_point, end_point, (255, 0, 0))
    _, masked_image = mask_image(masked_image, sp1, ep1)
    _, masked_image = mask_image(masked_image, sp2, ep2)
    _, masked_image = mask_image(masked_image, sp3, ep3)
    _, masked_image = mask_image(masked_image, sp4, ep4)
    _, masked_image = mask_image(masked_image, sp5, ep5)

    # Display images
    cv2.imshow("Image with Rectangles", image_with_rectangles)
    cv2.imshow("Masked Image", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
