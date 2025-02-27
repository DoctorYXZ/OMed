import os
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from swin_transformer import swin_tiny_patch4_window7_224 as create_model
from tqdm import tqdm

def load_image(img_path, data_transform):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    if not img_path.lower().endswith(valid_extensions):
        raise ValueError(f"Unsupported file format: {img_path}")

    try:
        img = Image.open(img_path).convert("RGB")
    except UnidentifiedImageError:
        raise UnidentifiedImageError(f"Cannot identify image file: {img_path}")

    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)  # expand batch dimension
    return img

def evaluate_model(model, device, data_transform, root_dir, class_names):
    wsi_labels = defaultdict(list)
    wsi_predictions = defaultdict(list)
    wsi_names_set = set()  # Use a set to maintain unique WSI names

    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Directory {class_dir} does not exist. Skipping.")
            continue

        img_names = os.listdir(class_dir)
        for img_name in tqdm(img_names, desc=f"Processing {class_name}"):
            # Skip Thumbs.db files
            if img_name == 'Thumbs.db':
                continue

            img_path = os.path.join(class_dir, img_name)
            wsi_name = img_name.split('.tif')[0] if '.tif' in img_name else '_'.join(img_name.split('_')[:2])

            # Ensure WSI name uniqueness
            if wsi_name not in wsi_names_set:
                wsi_names_set.add(wsi_name)

            try:
                img = load_image(img_path, data_transform).to(device)
            except (ValueError, UnidentifiedImageError) as e:
                print(e)
                continue

            with torch.no_grad():
                output = model(img).cpu()
                predict = torch.softmax(output, dim=1)
                score = predict[:, 1]

            wsi_predictions[wsi_name].append(score.item())
            wsi_labels[wsi_name].append(1 if class_name == 'RAS_Wildtype' else 0)    # The category needs to align with the training data. Assuming class 1 is 'RAS_Wildtype'.

    y_true = []
    y_score = []

    for wsi, scores in tqdm(wsi_predictions.items(), desc="Aggregating WSI scores"):
        final_score = sum(scores) / len(scores)  # Take the average score for the WSI
        majority_label = Counter(wsi_labels[wsi]).most_common(1)[0][0]  # Get the majority label

        y_score.append(final_score)
        y_true.append(majority_label)

    return y_true, y_score, list(wsi_names_set)

# Note: Since the trained model assigns Wildtype = 1, but we need to evaluate the model's performance in predicting Mutation, thus the sensitivity, specificity, PPV, and NPV need to be inverted, while the AUC and accuracy remain unchanged.
def compute_metrics(y_true, y_score, wsi_names):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Compute Youden's J statistic to determine the optimal threshold
    J = tpr - fpr
    ix = torch.argmax(torch.tensor(J))
    optimal_threshold = thresholds[ix]

    y_pred = [1 if score >= optimal_threshold else 0 for score in y_score]
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        raise ValueError("Confusion matrix is not 2x2. Check your input data and classes.")

    sensitivity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Note: Inverted
    specificity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Note: Inverted
    ppv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Note: Inverted
    npv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Note: Inverted
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0



    return {
        "AUC": roc_auc,
        "Optimal Threshold": optimal_threshold,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "Accuracy": accuracy
    }

def plot_roc_curve(y_true, y_score, output_dir, filename='roc_curve'):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(output_dir, f"{filename}.tif"), bbox_inches='tight', format='tif')
    plt.savefig(os.path.join(output_dir, f"{filename}.svg"), bbox_inches='tight', format='svg')
    plt.close()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.0313, 0.0086, 0.0286], [0.0978, 0.0530, 0.0939])   # The normalization parameters need to be consistent with the training dataset
    ])
    root_dir = 'C:\\path\\to\\your testing set'   # The testing set folder should contain two subfolders, RAS_Mutation and RAS_Wildtype, with each subfolder containing 224x224 pixel tumor patches
    class_names = ['RAS_Wildtype', 'RAS_Mutation']

    model = create_model(num_classes=2).to(device)
    model_weight_path = "C:\\code\\Mutation_prediction\\RAS\\weights\\best_model.pth"
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"Model weight file not found at {model_weight_path}")
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    model.eval()

    y_true, y_score, wsi_names = evaluate_model(model, device, data_transform, root_dir, class_names)

    if len(y_true) == 0:
        print("No valid samples found for evaluation.")
        return

    metrics = compute_metrics(y_true, y_score, wsi_names)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    plot_roc_curve(y_true, y_score, root_dir)

if __name__ == '__main__':
    main()