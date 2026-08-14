"""
KITTI object detection dataset loader.

KITTI label format (one object per line):
  type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y

We parse the type (Car, Pedestrian, Cyclist) and the 2D bounding box
coordinates (x1 y1 x2 y2 in pixel space).

Important: sequences are split at the sequence level, NOT the frame level.
Splitting by frame introduces data leakage because consecutive frames are
highly correlated — the model would see near-identical frames in both
train and test sets, artificially inflating evaluation metrics.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Classes we care about — ignore everything else in the KITTI labels
TARGET_CLASSES = ['Car', 'Pedestrian', 'Cyclist']
CLASS_TO_IDX = {cls: i for i, cls in enumerate(TARGET_CLASSES)}


def parse_label_file(label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a single KITTI label file.

    Returns:
      boxes:  (N, 4) float32 array of [x1, y1, x2, y2] in pixel coords
      labels: (N,)   int64  array of class indices
    """
    boxes, labels = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = parts[0]
            if cls not in CLASS_TO_IDX:
                continue
            x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            if x2 <= x1 or y2 <= y1:
                continue  # skip degenerate boxes
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_IDX[cls])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


class KITTIDataset(Dataset):
    """
    Custom PyTorch Dataset for KITTI object detection.

    Directory structure expected:
      kitti_root/
        images/   ← .png image files named XXXXXX.png
        labels/   ← .txt label files named XXXXXX.txt

    Args:
      kitti_root:    path to the dataset root
      frame_ids:     list of frame ID strings (e.g. ['000000', '000001'])
      transform:     optional image transforms (applied to PIL image)
      temporal_n:    if > 1, stack this many consecutive frames (temporal modelling)
    """

    def __init__(
        self,
        kitti_root: str,
        frame_ids: List[str],
        transform: Optional[transforms.Compose] = None,
        temporal_n: int = 1,
    ):
        self.root = Path(kitti_root)
        self.frame_ids = frame_ids
        self.temporal_n = temporal_n
        self.transform = transform or transforms.Compose([
            transforms.Resize((375, 1242)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.frame_ids)

    def _load_image(self, frame_id: str) -> torch.Tensor:
        img_path = self.root / 'images' / f'{frame_id}.png'
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

    def __getitem__(self, idx: int) -> Dict:
        frame_id = self.frame_ids[idx]
        boxes, labels = parse_label_file(self.root / 'labels' / f'{frame_id}.txt')

        if self.temporal_n > 1:
            # Stack N consecutive frames along the channel dimension
            # Frame N-1 is included if available, otherwise duplicate frame 0
            tensors = []
            for offset in range(-(self.temporal_n - 1), 1):
                neighbour_idx = max(0, idx + offset)
                neighbour_id = self.frame_ids[neighbour_idx]
                tensors.append(self._load_image(neighbour_id))
            image = torch.cat(tensors, dim=0)  # (3*N, H, W)
        else:
            image = self._load_image(frame_id)

        return {
            'image':    image,
            'boxes':    torch.tensor(boxes, dtype=torch.float32),
            'labels':   torch.tensor(labels, dtype=torch.long),
            'frame_id': frame_id,
        }
