import os
import numpy as np
import cv2
from PIL import Image
from skimage.morphology import disk, binary_opening

#overlap the 224*224 pixel patches with their visual heatmap
folder1_path = 'C:\\path\\to\\your color-normalized 224patch files'
folder2_path = 'C:\\path\\to\\visual heatmap for 224patches'
output_folder_path = 'C:\\path\\to\\save the overlapped 224patches'

# Create output folder (if it doesn't exist)
os.makedirs(output_folder_path, exist_ok=True)

# Go through all the.tif files in the first folder
for filename in os.listdir(folder1_path):
    if filename.endswith('.tif'):
        # Stitch the full path of the two pictures
        img1_path = os.path.join(folder1_path, filename)
        img2_path = os.path.join(folder2_path, filename)

        # Check if the corresponding image exists in the second folder
        if os.path.exists(img2_path):
            # Open the patches image
            img1 = Image.open(img1_path).convert('RGBA')
            img1_np = np.array(img1)[:, :, :3]  # Convert to three-channel RGB image

            # Apply Gaussian mode lines to smooth the image
            img1_blurred = cv2.GaussianBlur(img1_np, (5, 5), 0)

            # Convert to HSV color space
            hsv_image = cv2.cvtColor(img1_blurred, cv2.COLOR_RGB2HSV)

            # Threshold segmentation is performed using saturation channels because tissue areas tend to be more saturated
            s_channel = hsv_image[:, :, 1]

            # Use a lower threshold to capture more tissue areas
            threshold_value = 10
            _, tissue_mask = cv2.threshold(s_channel, threshold_value, 255, cv2.THRESH_BINARY)

            # Small organelle particles are removed by morphological manipulation and detailed tissue segments are preserved
            kernel = disk(10)
            opened_tissue_mask = binary_opening(tissue_mask, kernel)

            # Create a new image to overlay the heatmap
            blended = Image.new("RGBA", img1.size)

            # Overlay img2 onto img1
            img2 = Image.open(img2_path).convert('RGBA')
            for x in range(img2.width):
                for y in range(img2.height):
                    r1, g1, b1, a1 = img1.getpixel((x, y))  # Get the 224patch image pixels
                    r2, g2, b2, a2 = img2.getpixel((x, y))  # Get the heatmap pixels

                    # For background areas, the image pixels of 224patches are retained
                    if opened_tissue_mask[y, x] == 0:
                        blended.putpixel((x, y), (r1, g1, b1, a1))
                    else:
                        # For tissue areas, the heatmap pixels are overlapped
                        blended.putpixel((x, y), (r2, g2, b2, a2))

            # Save the overlapped images
            output_path = os.path.join(output_folder_path, filename)
            blended.save(output_path, format='TIFF')

print("The overlay is complete and the result has been saved to:", output_folder_path)
