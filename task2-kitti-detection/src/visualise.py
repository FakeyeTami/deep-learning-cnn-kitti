"""
Visualisation utilities for KITTI detection results.
Overlays ground-truth and predicted bounding boxes on images.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from typing import Dict, Optional


CLASS_NAMES  = ['Car', 'Pedestrian', 'Cyclist']
GT_COLOURS   = ['#22c55e', '#3b82f6', '#f59e0b']   # green, blue, amber — ground truth
PRED_COLOURS = ['#ef4444', '#8b5cf6', '#ec4899']   # red, purple, pink — predictions


def visualise_detections(
    image: torch.Tensor,
    ground_truth: Dict,
    predictions: Optional[Dict] = None,
    score_threshold: float = 0.5,
    title: str = '',
    save_path: Optional[str] = None,
) -> None:
    """
    Overlay bounding boxes on an image.

    Green/blue/amber = ground truth boxes
    Red/purple/pink  = predicted boxes (above score threshold)
    """
    # Denormalise image
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = image.permute(1, 2, 0).numpy()
    img  = (img * std + mean).clip(0, 1)

    fig, ax = plt.subplots(1, figsize=(14, 5))
    ax.imshow(img)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=11)

    # Ground-truth boxes
    for box, label in zip(ground_truth['boxes'], ground_truth['labels']):
        x1, y1, x2, y2 = box.tolist()
        colour = GT_COLOURS[int(label)]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=colour, facecolor='none', linestyle='solid'
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 3, f'GT: {CLASS_NAMES[int(label)]}',
                color=colour, fontsize=7, fontweight='bold')

    # Predicted boxes
    if predictions is not None:
        for box, label, score in zip(
            predictions['boxes'], predictions['labels'], predictions['scores']
        ):
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            colour = PRED_COLOURS[int(label)]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=colour, facecolor='none', linestyle='dashed'
            )
            ax.add_patch(rect)
            ax.text(x2, y1 - 3, f'{CLASS_NAMES[int(label)]} {score:.2f}',
                    color=colour, fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
