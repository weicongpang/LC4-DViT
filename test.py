import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets.folder import IMG_EXTENSIONS

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix,
    precision_recall_fscore_support
)

from models.build import build_model

# Add additional image extensions
IMG_EXTENSIONS += ('.tif', '.tiff')

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------- Configuration and File Path --------------------
BATCH_SIZE = 64
IM_SIZE = 512
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

TEST_DATASET_DIR = '/root/autodl-tmp/Processed_RS_del_all/test'
WEIGHTS_PATH = '/root/lvc2dvit/results/mobilenet_all/run_mobilenet_20251121_162703/checkpoint_final.pth'
CLASS_MAP_PATH = '/root/lvc2dvit/classification_to_name.json'
RESULTS_DIR = '/root/lvc2dvit/results/mobilenet_all/run_mobilenet_20251121_162703'

CM_FILENAME = 'confusion_matrix_mobilenet.png'
RESULTS_FILENAME = 'results.txt'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True


# -------------------- Data Preparation --------------------
def get_test_loader():
    test_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((IM_SIZE, IM_SIZE), InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_set = datasets.ImageFolder(TEST_DATASET_DIR, transform=test_transforms)
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return test_loader


def load_class_map():
    with open(CLASS_MAP_PATH, 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    return class_map


def load_model(num_classes):
    model = build_model().to(DEVICE)
    model.eval()

    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)

    print(f'Loaded weights from: {WEIGHTS_PATH}')
    return model


# -------------------- Evaluation --------------------
def evaluate(model, test_loader, class_map, results_dir):
    num_classes = len(class_map)

    y_true, y_pred = [], []
    total_per_class = [0] * num_classes
    correct_per_class = [0] * num_classes

    model.eval()
    start = time.time()

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            for t, p in zip(labels, preds):
                total_per_class[t] += 1
                correct_per_class[t] += int(t == p)

    elapsed = time.time() - start
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # -------------------- Confusion Matrix --------------------
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(font_scale=1.2)

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor='gray',
        xticklabels=[class_map.get(str(i), f'class_{i}') for i in range(num_classes)],
        yticklabels=[class_map.get(str(i), f'class_{i}') for i in range(num_classes)]
    )
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title('Confusion Matrix', fontsize=16, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(results_dir, CM_FILENAME)
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\nConfusion matrix saved as {cm_path}')

    # -------------------- Normalized Confusion Matrix --------------------
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(font_scale=1.2)

    ax2 = sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor='gray',
        xticklabels=[class_map.get(str(i), f'class_{i}') for i in range(num_classes)],
        yticklabels=[class_map.get(str(i), f'class_{i}') for i in range(num_classes)]
    )
    ax2.set_xlabel('Predicted Label', fontsize=14)
    ax2.set_ylabel('True Label', fontsize=14)
    ax2.set_title('Normalized Matrix Heatmap', fontsize=16, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_norm_path = os.path.join(results_dir, 'normalized_confusion_matrix_mobilenet.png')
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Normalized confusion matrix saved as {cm_norm_path}')


    oa = accuracy_score(y_true, y_pred)
    class_accs = [
        correct_per_class[i] / total_per_class[i]
        for i in range(num_classes) if total_per_class[i] > 0
    ]
    macc = np.mean(class_accs)
    kappa = cohen_kappa_score(y_true, y_pred)

    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division='warn')
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division='warn')
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division='warn')

    # -------------------- Per-class statistics --------------------
    per_class_stats = []
    N = cm.sum()
    for idx in range(num_classes):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = N - tp - fp - fn

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        po = (tp + tn) / N
        pe = (((tp + fn) * (tp + fp)) + ((fp + tn) * (fn + tn))) / (N * N)
        kappa_cls = (po - pe) / (1 - pe + 1e-8)

        per_class_stats.append({
            "idx": idx,
            "name": class_map.get(str(idx), f"class_{idx}"),
            "acc": (correct_per_class[idx] / total_per_class[idx]) * 100 if total_per_class[idx] > 0 else 0.0,
            "precision": precision,
            # "recall": recall,
            "f1": f1,
            "kappa": kappa_cls,
            "support": int(total_per_class[idx]),
        })

    # -------------------- Save results to txt --------------------
    results_path = os.path.join(results_dir, RESULTS_FILENAME)
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"Test completed. Samples: {len(y_true)}  Time: {elapsed:.2f}s\n")
        f.write(f"Overall Accuracy (OA) : {oa:.4f}\n")
        f.write(f"Mean Accuracy (mAcc)  : {macc:.4f}\n")
        f.write(f"Cohen-Kappa           : {kappa:.4f}\n")
        f.write(f"Precision (macro)     : {prec_macro:.4f}\n")
        f.write(f"Recall    (macro)     : {rec_macro:.4f}\n")
        f.write(f"F1-score  (macro)     : {f1_macro:.4f}\n\n")

        f.write("Idx  ClassName            Acc(%)   Prec        F1       Kappa    Support\n")
        f.write("-------------------------------------------------------------------------------\n")
        for cls in per_class_stats:
            f.write(
                f"[{cls['idx']:02d}] {cls['name']:<20s}: "
                f"{cls['acc']:6.2f}%  "
                f"{cls['precision']:7.4f}  "
                # f"{cls['recall']:7.4f}  "
                f"{cls['f1']:7.4f}  "
                f"{cls['kappa']:7.4f}  "
                f"{cls['support']:6d}\n"
            )


    print(f"\nResults saved as {results_path}")

    # -------------------- Print results to console --------------------
    print(f"\nTest completed. Samples: {len(y_true)}  Time: {elapsed:.2f}s")
    print(f"Overall Accuracy (OA) : {oa:.4f}")
    print(f"Mean Accuracy (mAcc)  : {macc:.4f}")
    print(f"Cohen-Kappa           : {kappa:.4f}")
    print(f"Precision (macro)     : {prec_macro:.4f}")
    print(f"Recall    (macro)     : {rec_macro:.4f}")
    print(f"F1-score  (macro)     : {f1_macro:.4f}")


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    test_loader = get_test_loader()
    class_map = load_class_map()
    model = load_model(len(class_map))

    transform = transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE), InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    evaluate(model, test_loader, class_map, RESULTS_DIR)
