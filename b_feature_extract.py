import cv2
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops
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

def extract_ball_patches(rgb_image_path, ground_image_path):
    # Load RGB image and corresponding GT mask
    rgb_image = cv2.imread(rgb_image_path)
    ground_image = cv2.imread(ground_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert GT mask to binary
    _, binary_mask = cv2.threshold(ground_image, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract patches based on contours
    ball_patches = []
    for contour in contours:
        # Get bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract ball patch from original image
        ball_patch = rgb_image[y:y+h, x:x+w]
        ball_patches.append(ball_patch)
    
    return ball_patches


def calculate_glcm_patches(ball_patches):
    glcm_patches = []
    for patch in ball_patches:
        # Convert patch to grayscale
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # Calculate GLCM for each color channel
        glcm_red = graycomatrix(gray_patch, [1], [0], levels=256, symmetric=True, normed=True)
        glcm_green = graycomatrix(gray_patch, [1], [0], levels=256, symmetric=True, normed=True)
        glcm_blue = graycomatrix(gray_patch, [1], [0], levels=256, symmetric=True, normed=True)
        
        # Append GLCMs to the list
        glcm_patches.append((glcm_red, glcm_green, glcm_blue))
    
    return glcm_patches


def compute_haralick_features(glcm):
    # Compute Haralick texture features
    asm = graycoprops(glcm, prop='ASM')
    contrast = graycoprops(glcm, prop='contrast')
    correlation = graycoprops(glcm, prop='correlation')
    
    return asm, contrast, correlation



# Function to calculate feature averages and ranges across orientations
def calculate_feature_stats(glcm_patches):
    avg_features = []
    range_features = []
    for glcm_red, glcm_green, glcm_blue in glcm_patches:
        # Compute features for each color channel
        asm_red, contrast_red, correlation_red = compute_haralick_features(glcm_red)
        asm_green, contrast_green, correlation_green = compute_haralick_features(glcm_green)
        asm_blue, contrast_blue, correlation_blue = compute_haralick_features(glcm_blue)
        
        # Calculate average and range for each feature
        avg_asm = np.mean([asm_red.mean(), asm_green.mean(), asm_blue.mean()])
        avg_contrast = np.mean([contrast_red.mean(), contrast_green.mean(), contrast_blue.mean()])
        avg_correlation = np.mean([correlation_red.mean(), correlation_green.mean(), correlation_blue.mean()])
        
        range_asm = np.ptp([asm_red, asm_green, asm_blue])
        range_contrast = np.ptp([contrast_red, contrast_green, contrast_blue])
        range_correlation = np.ptp([correlation_red, correlation_green, correlation_blue])
        
        # Append to lists
        avg_features.append((avg_asm, avg_contrast, avg_correlation))
        range_features.append((range_asm, range_contrast, range_correlation))
    
    return avg_features, range_features

# Function to select one feature from each color channel
def select_features(avg_features):
    # Select one feature from each color channel
    selected_features = [(avg_asm, avg_contrast, avg_correlation) for avg_asm, avg_contrast, avg_correlation in avg_features]
    
    return selected_features

# Function to plot the distribution of selected features per ball type
def plot_distribution(selected_features):
    feature_names = ['Average ASM', 'Average Contrast', 'Average Correlation']
    
    for i, feature_name in enumerate(feature_names):
        plt.figure(figsize=(10, 6))
        plt.title(f'Distribution of {feature_name} per Ball Type')
        for ball_type, features in enumerate(selected_features):
            print(f"Ball Type: {ball_type}, Features type: {type(features)}, Features: {features}")
            feature_values = features
            plt.hist(feature_values, bins=20, alpha=0.5, label=f'Ball Type {ball_type+1}')
        plt.xlabel(feature_name)
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()













rgb_image_paths = getImagePath("ball_frames/")
ground_image_paths = getGnImagePath("ball_frames/")

# Extract patches for the first image in the list
ball_patches = extract_ball_patches(rgb_image_paths[0], ground_image_paths[0])
glcm_patches = calculate_glcm_patches(ball_patches)
# Display the first extracted ball patch
cv2.imshow("Ball Patch", ball_patches[2])
avg_features, _ = calculate_feature_stats(glcm_patches)
selected_features = select_features(avg_features)
plot_distribution(selected_features)
cv2.waitKey(0)
cv2.destroyAllWindows()
