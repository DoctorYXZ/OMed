import torch
import numpy as np
import random
import os

# set random seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from datasets import TransUNetDataset
from networks.transunet import TransUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Dataset paths
train_dir = 'C:\\path\\to\\your training set'   # The training set should contain both .tif format image and .json format annotation of tumor region for each 896*896 patch image
save_file = 'C:\\models\\epochs\\transunet_best.pth'  # Path to save the best model
epoch_save_dir = 'C:\\models\\epochs\\'  # Path to save each epoch's model
os.makedirs(epoch_save_dir, exist_ok=True)  # Create directory (if it doesn't exist)

# Create dataset instance
train_dataset = TransUNetDataset(train_dir, is_train=True)

# Data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

# Create TransUNet instance
unet = TransUNet(img_dim=896,
                 in_channels=3,
                 out_channels=128,
                 head_num=4,
                 mlp_dim=512,
                 block_num=8,
                 patch_dim=16,
                 class_num=1
                 )

# Wrap the model in DataParallel and move it to the available GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    unet = nn.DataParallel(unet)

unet.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(unet.parameters(), lr=0.001, weight_decay=1e-4)  # Initial learning rate and L2 regularization

# Training epochs
num_epochs = 100

# Training process
unet.train()
less_loss = 1000000
loss_history = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    epoch_loss = 0.0

    for images, masks, _ in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = unet(images)
        loss = criterion(outputs, masks.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Loss: {epoch_loss / len(train_loader)}")
    loss_history.append(epoch_loss / len(train_loader))

    # Save the model for each epoch
    torch.save(unet.module.state_dict() if isinstance(unet, nn.DataParallel) else unet.state_dict(),
               os.path.join(epoch_save_dir, f'transunet_epoch_{epoch + 1}.pth'))

    # If the current epoch's loss is lower than the previous minimum loss, save the best model
    if epoch_loss < less_loss:
        less_loss = epoch_loss
        torch.save(unet.module.state_dict() if isinstance(unet, nn.DataParallel) else unet.state_dict(), save_file)
