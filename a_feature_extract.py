import cv2
import os
import numpy as np
from skimage import io, measure
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


solidity_values = []
non_compactness_values = []
circularity_values = []
eccentricity_values = []
min_region_size = 100

ball_type_shape_features = {}
rgb_image_path = getImagePath("ball_frames/")
ground_image_path = getGnImagePath("ball_frames/")
feature_names = ['Solidity', 'Non-Compactness', 'Circularity', 'Eccentricity']
for idx,img in enumerate(rgb_image_path):
    ground_image = cv2.imread(ground_image_path[idx])
    for slice in ground_image:
        labeled_slice = measure.label(slice)
        
        for region in measure.regionprops(labeled_slice):
            if region.area >= min_region_size:
                solidity = region.solidity
                non_compactness = region.perimeter / np.sqrt(region.area)
                if region.perimeter != 0:
                    circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
                else:
                    circularity = 0
                eccentricity = region.eccentricity
                solidity_values.append(solidity)
                non_compactness_values.append(non_compactness)
                circularity_values.append(circularity)
                eccentricity_values.append(eccentricity)


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.hist(solidity_values, bins=20)
plt.title('Solidity Distribution')
plt.xlabel('Solidity')

plt.subplot(2, 2, 2)
plt.hist(non_compactness_values, bins=20)
plt.title('Non-Compactness Distribution')
plt.xlabel('Non-Compactness')

plt.subplot(2, 2, 3)
plt.hist(circularity_values, bins=20)
plt.title('Circularity Distribution')
plt.xlabel('Circularity')

plt.subplot(2, 2, 4)
plt.hist(eccentricity_values, bins=20)
plt.title('Eccentricity Distribution')
plt.xlabel('Eccentricity')

plt.tight_layout()
plt.show()
