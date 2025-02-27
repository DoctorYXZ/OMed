import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from loss import *

def read_data_from_two_sets(train_root: str, val_root: str):
    print(f"Training dataset root: {train_root}")
    print(f"Validation dataset root: {val_root}")
    assert os.path.exists(train_root), f"Training dataset root: {train_root} does not exist."
    assert os.path.exists(val_root), f"Validation dataset root: {val_root} does not exist."

    # Helper function to read dataset paths and labels
    def read_dataset(root: str):
        gene_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
        gene_class.sort()
        class_indices = dict((k, v) for v, k in enumerate(gene_class))
        images_path = []
        images_label = []
        supported = [".jpg", ".JPG", ".png", ".PNG", ".tiff", ".TIFF", ".tif"]

        for cla in gene_class:
            cla_path = os.path.join(root, cla)
            images = [os.path.join(root, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
            images.sort()
            images_path += images
            images_label += [class_indices[cla]] * len(images)

        return images_path, images_label

    train_images_path, train_images_label = read_dataset(train_root)
    val_images_path, val_images_label = read_dataset(val_root)

    return train_images_path, train_images_label, val_images_path, val_images_label



def read_data_and_split(root: str, test_size=0.2, random_state=42):
    print(f"Dataset root: {root}")
    assert os.path.exists(root), f"Dataset root: {root} does not exist."

    gene_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    gene_class.sort()
    class_indices = {k: v for v, k in enumerate(gene_class)}
    
    images_path = []
    images_label = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".tiff", ".TIFF", ".tif"]

    for cla in gene_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
        images.sort()
        images_path += images
        images_label += [class_indices[cla]] * len(images)

    train_images_path, val_images_path, train_images_label, val_images_label = train_test_split(
        images_path, images_label, test_size=test_size, random_state=random_state, stratify=images_label
    )

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # Reverse normalization
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # Remove x-axis ticks
            plt.yticks([])  # Remove y-axis ticks
            plt.imshow(img.astype('uint8'))
        plt.show()

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)

def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = FocalLoss()
    accu_loss = torch.zeros(1).to(device)  # Accumulated loss
    accu_num = torch.zeros(1).to(device)   # Accumulated correct predictions count
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data

        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = FocalLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # Accumulated correct predictions count
    accu_loss = torch.zeros(1).to(device)  # Accumulated loss

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
