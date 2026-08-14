"""
Evaluation utilities for MNIST classification.
Computes per-class metrics and generates confusion matrix.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from typing import Tuple


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (all_preds, all_labels) arrays."""
    model.eval()
    preds, labels = [], []
    for images, lbls in loader:
        outputs = model(images.to(device))
        preds.extend(outputs.argmax(1).cpu().numpy())
        labels.extend(lbls.numpy())
    return np.array(preds), np.array(labels)


def print_classification_report(preds: np.ndarray, labels: np.ndarray) -> None:
    class_names = [str(i) for i in range(10)]
    print(classification_report(labels, preds, target_names=class_names))


def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray, save_path: str = None) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion matrix — MNIST')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
