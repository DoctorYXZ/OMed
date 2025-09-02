import os
from PIL import Image

def create_wsi_from_patches(patch_directory, output_image_path):
    patches = {}
    max_x, max_y = 0, 0

    # Read each patch and store it in the dictionary
    for filename in os.listdir(patch_directory):
        if filename.endswith('.tif'):
            parts = filename.split('_')
            x = int(parts[-2])
            y = int(parts[-1].replace('.tif', ''))

            # Open the patch using with statement to ensure it closes properly
            patch_path = os.path.join(patch_directory, filename)
            with Image.open(patch_path) as patch:
                patches[(x, y)] = patch.copy()  # Copy the image to keep it outside the with block

            # Update the max coordinates
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    # Determine patch size
    patch_size = list(patches.values())[0].size

    # Create the full WSI canvas
    wsi_width = max_x + patch_size[0]
    wsi_height = max_y + patch_size[1]
    wsi_image = Image.new('RGB', (wsi_width, wsi_height))

    # Paste each patch into the WSI image
    for (x, y), patch in patches.items():
        wsi_image.paste(patch, (x, y))

    # Save the full WSI image
    wsi_image.save(output_image_path)


patch_directory = 'C:\\path\\to\\your overlapped 224patches'
output_image_path = 'C:\\path\\to\\save\\WSI_heatmap.tif'
create_wsi_from_patches(patch_directory, output_image_path)
