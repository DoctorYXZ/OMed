import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from networks.transunet import TransUNet
from datasets import TransUNetDataset
from collections import OrderedDict
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

test_dir = 'C:\\path\\to\\your dataset for tumor region detection'  # The dataset should contain .tif 896*896 patch images
test_dataset = TransUNetDataset(test_dir, is_train=False)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# Initialize model
unet = TransUNet(img_dim=896,
                 in_channels=3,
                 out_channels=128,
                 head_num=4,
                 mlp_dim=512,
                 block_num=8,
                 patch_dim=16,
                 class_num=1,
                 )


if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    unet = nn.DataParallel(unet)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet.to(device)

# Load pretrained model
pretrained_dict = torch.load('C:\\models\\epochs\\transunet_best.pth or transunet_epoch_100.pth')

#  If the model was previously trained on multiple GPUs, the following Settings are required to run the model on a single GPU without reporting errors
if not isinstance(unet, nn.DataParallel) and any(k.startswith('module.') for k in pretrained_dict.keys()):
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:]  # Remove the 'module.' prefix
        new_state_dict[name] = v
    unet.load_state_dict(new_state_dict)
else:
    unet.load_state_dict(pretrained_dict)

save_path = 'C:\\path\\to\\save your detection results'
os.makedirs(save_path, exist_ok=True)

for i in range(len(test_dataset)):
    unet.eval()

    image, true_mask, filename = test_dataset[i]
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = unet(image)

    predicted_mask = torch.sigmoid(output) > 0.5
    predicted_mask = predicted_mask.type(torch.uint8)
    predicted_mask = predicted_mask.squeeze(0).cpu().numpy()
    predicted_mask = predicted_mask.reshape(896, 896)

    original_image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    masked_image = original_image_np * np.stack([predicted_mask, predicted_mask, predicted_mask], axis=-1)
    masked_image = np.clip(masked_image, 0, 1)

    plt.imsave(f'{save_path}/{filename}_masked_image.tiff', masked_image)
    plt.close()