import os
import math
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np

from my_dataset import MyDataSet
from swin_transformer import swin_tiny_patch4_window7_224 as create_model
from utils import train_one_epoch, evaluate, read_data_and_split

tb_writer = SummaryWriter(log_dir='./logs')


def compute_class_accuracy(model, data_loader, device):
    model.eval()
    class_correct = torch.zeros(args.num_classes).to(device)
    class_total = torch.zeros(args.num_classes).to(device)
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).float()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i]
                class_total[label] += 1
    
    class_acc = (class_correct / class_total).cpu().numpy()
    return class_acc


def evaluate_and_plot_roc(model, data_loader, device, dataset_name, epoch, save_path):
    model.eval()
    y_real = []
    y_proba = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probas = torch.softmax(outputs, dim=1)[:, 1]
            y_real.extend(labels.cpu().numpy())
            y_proba.extend(probas.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(y_real, y_proba)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    y_pred = [1 if x >= optimal_threshold else 0 for x in y_proba]
    cm = confusion_matrix(y_real, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {dataset_name}')
    plt.legend(loc="lower right")
    tif_path = os.path.join(save_path, f"roc_curve_{dataset_name}_epoch_{epoch}.tif")
    svg_path = os.path.join(save_path, f"roc_curve_{dataset_name}_epoch_{epoch}.svg")
    plt.savefig(tif_path)
    plt.savefig(svg_path)
    plt.close()
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print(f"AUC: {roc_auc:.3f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}, PPV: {ppv:.3f}, NPV: {npv:.3f}")

    return tif_path, svg_path, roc_auc, sensitivity, specificity, ppv, npv

def compute_mean_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    return mean, std

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weights_dir = os.path.join(os.path.dirname(args.data_path), 'weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    tb_writer = SummaryWriter()

    # Use the data reading and splitting function
    train_images_path, train_images_label, val_images_path, val_images_label = read_data_and_split(args.data_path)

    print("Training images and labels (first 5):", train_images_path[:5], train_images_label[:5])
    print("Validation images and labels (first 5):", val_images_path[:5], val_images_label[:5])

    # Initial transformation, only ToTensor
    initial_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Create initial dataset and data loader
    initial_dataset = MyDataSet(images_path=train_images_path, images_class=train_images_label, transform=initial_transform)
    initial_loader = DataLoader(initial_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

    # Calculate mean and standard deviation
    mean, std = compute_mean_std(initial_loader)
    print(f"Calculated mean: {mean}, std: {std}")

    # Define the final data transformation using the calculated mean and standard deviation
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    # Create training and validation datasets using the final data transformation
    train_dataset = MyDataSet(images_path=train_images_path, images_class=train_images_label, transform=data_transform["train"])
    val_dataset = MyDataSet(images_path=val_images_path, images_class=val_images_label, transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # Adjust according to your system
    print(f'Using {nw} dataloader workers every process')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw, collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # Remove the pre-trained model's weights related to classification categories
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    # If use multi-GPU training
    #model = create_model(num_classes=args.num_classes).to(device)
    #if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model)


    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lrf)

    model.dropout = torch.nn.Dropout(p=0.5)

    best_val_auc = 0
    patience = 6  # The early stopping parameter can be adjusted as needed
    patience_counter = 0

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                                device=device, epoch=epoch)
        
        # Validation
        model.eval()
        val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)
        
        # Compute class-wise accuracy for training and validation sets
        train_class_acc = compute_class_accuracy(model, train_loader, device)
        val_class_acc = compute_class_accuracy(model, val_loader, device)
        
        # Log metrics
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print("Train Class Accuracies:", train_class_acc)
        print("Val Class Accuracies:", val_class_acc)
        
        # TensorBoard logging
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_acc", train_acc, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_acc", val_acc, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        
        for i, acc in enumerate(train_class_acc):
            tb_writer.add_scalar(f"train_class_{i}_acc", acc, epoch)
        for i, acc in enumerate(val_class_acc):
            tb_writer.add_scalar(f"val_class_{i}_acc", acc, epoch)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save model
        model_save_path = os.path.join(weights_dir, f"weights_model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        
        # Compute and log ROC curve and AUC at the end of each epoch
        train_tif_path, train_svg_path, train_auc, train_sensitivity, train_specificity, train_ppv, train_npv = evaluate_and_plot_roc(
            model, train_loader, device, "Training Set", epoch, args.data_path)
        val_tif_path, val_svg_path, val_auc, val_sensitivity, val_specificity, val_ppv, val_npv = evaluate_and_plot_roc(
            model, val_loader, device, "Validation Set", epoch, args.data_path)
        
        print(f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        tb_writer.add_scalar("train_auc", train_auc, epoch)
        tb_writer.add_scalar("val_auc", val_auc, epoch)
        
        # Early stopping (optional)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(weights_dir, "best_model.pth"))
            print(f"New best model saved with Val AUC: {best_val_auc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered. No improvement for {patience} epochs.")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weights', type=str, default='C:\\code\\Mutation_prediction\\RAS\\swin_tiny_patch4_window7_224.pth',
                        help='initial pre-trained weights path')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data_path', type=str, default='C:\\path\\to\\your training set')   # The training set folder should contain two subfolders, RAS_Mutation and RAS_Wildtype, with each subfolder containing 224x224 pixel tumor patches
    parser.add_argument('--freeze_layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()
    main(args)