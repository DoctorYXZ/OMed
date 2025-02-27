from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import json
import cv2

Image.MAX_IMAGE_PIXELS = None  # Prevent exceptions when images are too large

class TransUNetDataset(Dataset):
    """
    A class for creating datasets that can handle tiff images and json files
    """
    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        if is_train:
            # For the training set, load json files
            self.json_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.json')]
        else:
            # For the test set, load tif image files
            self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.json_files) if self.is_train else len(self.image_files)

    def __getitem__(self, idx):
        if self.is_train:
            # Process the training set
            json_file = self.json_files[idx]
            with open(json_file, 'r') as file:
                json_data = json.load(file)

            image_path = os.path.join(self.root_dir, json_data['imagePath'].split('\\')[-1])
            labels = json_data['shapes']

            image = Image.open(image_path)
            width, height = image.size

            mask = np.zeros((height, width), dtype=np.uint8)

            for label in labels:
                points = label['points']
                polygon = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [polygon], 1)

            transform = transforms.Compose([
                transforms.Resize((896, 896)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

            image = transform(image)

            mask = Image.fromarray(mask)
            mask = mask.resize((896, 896))
            mask = np.array(mask)
            mask = torch.from_numpy(mask)

            filename = os.path.basename(image_path)
            return image, mask, filename
        else:
            # Process the test set
            image_path = self.image_files[idx]
            image = Image.open(image_path)

            transform = transforms.Compose([
                transforms.Resize((896, 896)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

            image = transform(image)
            filename = os.path.basename(image_path)

            # There is no mask in the test set, so return None as a placeholder for the mask
            return image, None, filename


