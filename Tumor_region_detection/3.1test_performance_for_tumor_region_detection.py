import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from networks.transunet import TransUNet
from datasets import TransUNetDataset
from collections import OrderedDict
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Setting environment variable to handle OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


test_dir = 'C:\\path\\to\\your testing set'  # The testing set should contain both .tif and .json for each evaluated 896*896 patch image
results_dir = 'C:\\path\\to\\save your testing results'
os.makedirs(results_dir, exist_ok=True)

test_dataset = TransUNetDataset(test_dir)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# Initialize model
unet = TransUNet(img_dim=896,
                 in_channels=3,
                 out_channels=128,
                 class_num=1,
                 head_num=4,
                 mlp_dim=512,
                 block_num=8,
                 patch_dim=16)

# Check for multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    unet = nn.DataParallel(unet)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet.to(device)

# Load pretrained model
pretrained_dict = torch.load('C:\\models\\epochs\\transunet_best.pth or transunet_epoch_100.pth',
                             map_location=device)

#  If the model was previously trained on multiple GPUs, the following Settings are required to run the model on a single GPU without reporting errors
if not isinstance(unet, nn.DataParallel) and any(k.startswith('module.') for k in pretrained_dict.keys()):
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:]  # Remove the 'module.' prefix
        new_state_dict[name] = v
    unet.load_state_dict(new_state_dict)
else:
    unet.load_state_dict(pretrained_dict)

# Initialize lists for ROC curve and accuracy
all_labels = []
all_preds = []
accuracies = []

unet.eval()
with torch.no_grad():
    for idx, (image, true_mask, filenames) in enumerate(test_loader):
        image = image.to(device)
        true_mask = true_mask.to(device)
        filename = filenames[0]

        output = unet(image)
        predicted_probs = torch.sigmoid(output).squeeze().cpu().numpy().flatten()

        # Predict labels based on a 0.5 threshold
        predicted_labels = (predicted_probs > 0.5).astype(np.float32)

        # Update lists
        all_preds.extend(predicted_probs)
        all_labels.extend(true_mask.squeeze().cpu().numpy().flatten())

        # Calculate batch accuracy and add to list
        accuracy = (predicted_labels == true_mask.squeeze().cpu().numpy().flatten()).mean()
        accuracies.append(accuracy)

        # Save images and masks using filename as part of the file name
        save_image(image.cpu(), os.path.join(results_dir, f'{filename}_image.tif'))
        save_image(torch.tensor(predicted_labels, device=device).reshape(true_mask.shape).float().cpu(),
                   os.path.join(results_dir, f'{filename}_predicted_mask.tif'))
        save_image(true_mask.float().cpu(), os.path.join(results_dir, f'{filename}_true_mask.tif'))

        # Generate and save masked_image
        mask_image = image.clone()
        predicted_labels_tensor = torch.tensor(predicted_labels, device=device).reshape(true_mask.shape).float()
        for c in range(image.shape[1]):  # For each channel
            mask_image[:, c, :] = image[:, c, :] * predicted_labels_tensor
        save_image(mask_image.cpu(), os.path.join(results_dir, f'{filename}_mask_image.tif'))

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

# Calculate sensitivity, specificity, PPV, and NPV
tn, fp, fn, tp = confusion_matrix(all_labels, (np.array(all_preds) > 0.5).astype(int)).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp)
npv = tn / (tn + fn)

# Calculate the mean accuracy
mean_accuracy = np.mean(accuracies)

# Display calculated metrics
print(f"ROC AUC: {roc_auc}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"PPV: {ppv}")
print(f"NPV: {npv}")
print(f"Mean Accuracy: {mean_accuracy}")

# Plot and save ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) of TransUNet')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, 'ROC_Curve_train_transunet.svg'))
plt.savefig(os.path.join(results_dir, 'ROC_Curve_train_transunet.tif'))
plt.close()

# Plot accuracy per sample and mean accuracy
plt.figure()
plt.plot(accuracies, label='Accuracy per Sample')
plt.axhline(y=mean_accuracy, color='r', linestyle='--', label=f'Mean Accuracy: {mean_accuracy:.2f}')
plt.xlabel('Sample')
plt.ylabel('Accuracy')
plt.title('Test Sample Accuracy and Mean Accuracy')
plt.legend()
plt.savefig(os.path.join(results_dir, 'Mean_Accuracy_Visualization.tif'))
plt.close()
