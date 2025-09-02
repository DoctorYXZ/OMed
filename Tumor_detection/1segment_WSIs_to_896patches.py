import os
# Set environment variables to avoid conflicts with the OpenMP runtime library.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.add_dll_directory('C:\\code\\Tumor_region_detection\\openslide-bin-4.0.0.2-windows-x64\\bin')

import cv2
import openslide
import numpy as np
from skimage.morphology import disk, binary_opening
from PIL import Image


# Define the folder path for the WSI files.
folder_path = 'C:\\path\\to\\your WSI files'
# Define the folder path for the output 896*896 pixel patches.
sub_output_folder_path = 'C:\\path\\to\\save your 896patch files'
# If the output folder does not exist, create it
if not os.path.exists(sub_output_folder_path):
    os.makedirs(sub_output_folder_path)

# Go through all the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.tif'):  # Check if the file is in.tif format
        slide_path = os.path.join(folder_path, filename)
        slide = openslide.OpenSlide(slide_path)

        # Read the image in original resolution
        wsi_image = slide.read_region((0, 0), 0, slide.level_dimensions[0])
        wsi_image = np.array(wsi_image)[:, :, :3]  # Delete the alpha channel

        # Segment the WSI images to 896*896patches and filter
        height, width, _ = wsi_image.shape
        for y in range(0, height, 896):
            for x in range(0, width, 896):
                if (x + 896 <= width) and (y + 896 <= height):
                    patch = wsi_image[y:y + 896, x:x + 896]

                    # Convert to HSV color space to check saturation
                    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
                    s_channel = hsv_patch[:, :, 1]

                    # Threshold segmentation using saturation channels
                    threshold_value = 10
                    _, tissue_mask = cv2.threshold(s_channel, threshold_value, 255, cv2.THRESH_BINARY)

                    # Calculate the proportion of tissue in the mask
                    tissue_area = np.sum(tissue_mask > 0)  # Calculate the number of tissue pixels in the mask
                    total_area = 896 * 896  # Total number of pixels in the current patch
                    tissue_ratio = tissue_area / total_area  # Calculate the proportion of tissue

                    # If the proportion exceeds 10%, save the patch
                    # This threshold can be adjusted according to your needs
                    if tissue_ratio > 0.10:
                        sub_image_pil = Image.fromarray(patch)

                        # Define the output path and save the segmented patches
                        sub_output_path = os.path.join(sub_output_folder_path, f'{filename}_{x}_{y}.tif')
                        sub_image_pil.save(sub_output_path)


        # Close the WSI file
        slide.close()
