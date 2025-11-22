import os
import warnings
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS
IMG_EXTENSIONS += ('.tif', '.tiff')

from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import seaborn as sns
from models.build import build_model
from torchinfo import summary
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from datetime import datetime
import logging

# Suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_EPOCHS = 30           
LR = 1e-4                
IM_SIZE = 512
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

TRAIN_DATASET_DIR = '/root/autodl-tmp/Processed_RS_del_all/train'
VALID_DATASET_DIR = '/root/autodl-tmp/Processed_RS_del_all/val'
RESULTS_ROOT = '/root/lvc2dvit/results/dvit_all_deleteall'
CHECKPOINT_INTERVAL = 15   

def init_logger(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def get_data_loaders():
    """
    保持你们现在的策略：不裁剪，不强增广。
    """
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((IM_SIZE, IM_SIZE), InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((IM_SIZE, IM_SIZE), InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.ImageFolder(TRAIN_DATASET_DIR, train_transform)
    val_ds   = datasets.ImageFolder(VALID_DATASET_DIR, val_transform)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader


def get_model():
    model = build_model().to(DEVICE)
    return model

def get_criterion():
    return nn.CrossEntropyLoss()

def get_optimizer(model):
    return AdamW(model.parameters(), lr=LR)

# ===================== Train & Val ===================== #
def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss, running_correct = 0.0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_correct / len(train_loader.dataset) * 100
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    model.eval()
    running_loss, running_correct = 0.0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = running_correct / len(val_loader.dataset) * 100
    return val_loss, val_acc

# ===================== Plot ===================== #
def plot_curves(history, out_path="curves.png"):
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": 200,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2.2
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], label="Validation Accuracy")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.replace(".png", ".svg"))
    plt.close()


if __name__ == "__main__":
    torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("run_dvit_all_deleteall_%Y%m%d_%H%M%S")
    result_dir = os.path.join(RESULTS_ROOT, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, "train_log.txt")
    logger = init_logger(log_path)
    logger.info("Logger initialized.")

    train_loader, val_loader = get_data_loaders()
    model = get_model()
    logger.info("Model Summary:")
    summary(model, input_size=(1, 3, IM_SIZE, IM_SIZE), device=str(DEVICE))

    criterion = get_criterion()
    optimizer = get_optimizer(model)

    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0       # ★ early stopping 计数
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    log_file = open(log_path, "w")

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        log_str = (
            f"Epoch {epoch:03d} | "
            f"Train Accuracy: {tr_acc:.2f}%, Train Loss: {tr_loss:.4f} | "
            f"Val Accuracy: {val_acc:.2f}%, Val Loss: {val_loss:.4f} | "
            f"Time: {dt:.1f}s"
        )
        logger.info(log_str)
        log_file.write(log_str + "\n")
        log_file.flush()

        # best model & early stopping
        if val_acc > best_acc + 1e-4:   
            best_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0

            best_model_path = os.path.join(result_dir, "best_dvit.pth")
            torch.save({
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "epoch": epoch
            }, best_model_path)
            best_info = f"*** New best Acc@1 {best_acc:.2f}% @ epoch {epoch} ***"
            logger.info(best_info)
            log_file.write(best_info + "\n")
            log_file.flush()
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epoch(s).")


        # Save checkpoint only at the last epoch
        if (epoch + 1) == NUM_EPOCHS:
            ckpt_path = os.path.join(result_dir, f"checkpoint_final.pth")
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "history": history
            }, ckpt_path)
            ckpt_info = f"Checkpoint saved at epoch {epoch+1}"
            logger.info(ckpt_info)
            log_file.write(ckpt_info + "\n")
            log_file.flush()

    final_info = f"\nFinished. Best Acc@1 {best_acc:.2f}% @ epoch {best_epoch}"
    logger.info(final_info)
    log_file.write(final_info + "\n")
    log_file.close()

    curve_path = os.path.join(result_dir, "curves.png")
    plot_curves(history, out_path=curve_path)
    logger.info(f"Curves saved at {curve_path}")
