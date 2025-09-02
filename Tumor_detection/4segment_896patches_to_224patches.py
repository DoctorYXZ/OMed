from PIL import Image
import numpy as np
import os

# This step segments the 896 patches containing tumor regions into 224 patches. Generally, the size of the tumor regions in the 224 patches should be sufficient for the subsequent mutation prediction model to capture adequate features, but they should not cover the entire image to avoid losing contextual information. The specific proportion can be adjusted based on the task requirements, but a common practice is for the target content to occupy 20% to 80% of the entire image area.
def is_colorful(patch, threshold=0.50):
    patch_np = np.array(patch)
    # Calculate the number of tumor regions' colorful pixels (here simplified as pixels with brightness above a certain threshold, assuming a brightness threshold of 70, which can be adjusted based on actual conditions)
    bright_pixels = np.sum(np.mean(patch_np, axis=-1) > 70)
    total_pixels = patch_np.shape[0] * patch_np.shape[1]
    colorful_ratio = bright_pixels / total_pixels
    return colorful_ratio > threshold

def save_colorful_patches(image_path, patch_size=224, threshold=0.50, save_dir='colorful_patches'):
    for filename in os.listdir(image_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
            continue
        img_path = os.path.join(image_path, filename)
        img = Image.open(img_path)
        os.makedirs(save_dir, exist_ok=True)
        width, height = img.size
        for x in range(0, width, patch_size):
            for y in range(0, height, patch_size):
                patch = img.crop((x, y, x + patch_size, y + patch_size))
                if is_colorful(patch, threshold):
                    patch_name = f'{filename[:-4]}_patch_{x}_{y}.tif'
                    patch.save(os.path.join(save_dir, patch_name))


image_path = 'C:\\path\\to\\the previous detection results'  # the 896*896 pixel patches containing tumor regions
patches_save_dir = 'C:\\path\\to\\save your segmentation results'  # the 224*224 pixel patches containing tumor regions
save_colorful_patches(image_path, save_dir=patches_save_dir)
