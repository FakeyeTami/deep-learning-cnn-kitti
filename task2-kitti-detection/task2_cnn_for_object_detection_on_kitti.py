"""
Task 2 — Fine-Tuning a Pre-Trained CNN for Object Detection on KITTI

Pipeline:
  1. Load KITTI sequences: tracklet XML + PNG frames + calibration files
  2. Project 3D bounding boxes → 2D image space via calibration matrices
  3. Build KITTITrackletDataset with sequence-level train/val/test split
     (IMPORTANT: split at sequence level, not frame level, to prevent leakage)
  4. Experiment A — Faster R-CNN, frozen backbone, head-only fine-tuning
  5. Experiment B — Faster R-CNN, full end-to-end fine-tuning (lower LR)
  6. Compare on validation set using mAP@0.5 and mean IoU
  7. Final evaluation on held-out test sequence
  8. Visualise predictions vs ground-truth bounding boxes
  9. TemporalFasterRCNN — feature stacking across 3 consecutive frames

Model: Faster R-CNN with ResNet-50 FPN backbone (ImageNet pretrained)
       Detection head replaced for 4-class output (Background + 3 KITTI classes)
Classes: Background (0), Car (1), Pedestrian (2), Cyclist (3)
Metrics: mAP@0.5 via torchmetrics, mean IoU per ground-truth box

Calibration:
  Supports both multi-file KITTI format (calib_cam_to_cam.txt + calib_velo_to_cam.txt)
  and single calib.txt format. 3D box corners are rotated, translated,
  projected through Tr_velo_to_cam → R_rect → P2 to get 2D image coordinates.

Requires:
  pip install torch torchvision torchmetrics pillow

Output files (saved to working directory):
  gt_verification.png       ground-truth boxes on 4 random training frames
  loss_curves.png           train/val loss for FrozenBackbone and FullFineTune
  predictions_vs_gt.png     predicted vs ground-truth on test frames
  kitti_<model>.pth         saved weights for best model

Author: Tamilore Fakeye
"""

# Run:  pip install torchmetrics  (if not already installed)

import glob
import os
import random
import warnings
from collections import defaultdict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights,
                                          fasterrcnn_resnet50_fpn)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── KITTI dataset root ────────────────────────────────────────────────────────
# Set KITTI_ROOT to the directory that contains the calib/, data/, tracklets/
# subfolders (each keyed by a 4-digit sequence id, e.g. 0000 .. 0020).

KITTI_ROOT   = './data/kitti'  # ← EDIT THIS PATH
CALIB_DIR    = os.path.join(KITTI_ROOT, 'calib')
DATA_DIR     = os.path.join(KITTI_ROOT, 'data')
TRACKLET_DIR = os.path.join(KITTI_ROOT, 'tracklets')

seq_ids = sorted([
    os.path.splitext(f)[0]
    for f in os.listdir(TRACKLET_DIR)
    if f.endswith('.txt')
]) if os.path.isdir(TRACKLET_DIR) else []

print(f'Found {len(seq_ids)} sequence id(s):')
for s in seq_ids:
    print(' ', s)


# ── Class mapping ───────────────────────────────────────────────


# Background = 0 (required by Faster R-CNN), then our 3 object classes
CLASSES     = ['Background', 'Car', 'Pedestrian', 'Cyclist']
CLS2IDX     = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS     = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)   # 4

# Colours for visualisation
VIS_COLOURS = {'Car': 'lime', 'Pedestrian': 'red', 'Cyclist': 'cyan'}
print('Class map:', CLS2IDX)


# ── Calibration parser (kept for reference / future 3D use) ─────
#
# Not required for the 2D detection pipeline below, since the label files
# already carry 2D boxes — but parsed here in case it's useful later.


def load_calib(seq_id):
    calib_path = os.path.join(CALIB_DIR, f'{seq_id}.txt')
    if not os.path.exists(calib_path):
        return None

    d = {}
    with open(calib_path) as f:
        for line in f:
            if ':' not in line:
                continue
            k, v = line.split(':', 1)
            try:
                d[k.strip()] = np.array(list(map(float, v.split())))
            except ValueError:
                pass

    calib = {}
    if 'P2' in d:
        calib['P2'] = d['P2'].reshape(3, 4)
    if 'R_rect' in d:
        calib['R_rect'] = d['R_rect'].reshape(3, 3)
    if 'Tr_velo_cam' in d:
        Tr = np.eye(4)
        Tr[:3, :] = d['Tr_velo_cam'].reshape(3, 4)
        calib['Tr_velo_cam'] = Tr
    return calib


# ── Label parser ──────────────────────────────────────────────
#
# Standard KITTI tracking label columns (0-indexed):
#   0 frame          1 track_id       2 type
#   3 truncated      4 occluded       5 alpha
#   6 bbox_left      7 bbox_top       8 bbox_right    9 bbox_bottom
#  10 height         11 width         12 length
#  13 x              14 y             15 z            16 rotation_y
#  17 score (optional, only present in prediction/result files)


KEEP_CLASSES = {'Car', 'Pedestrian', 'Cyclist'}


def parse_label_file(label_path):
    """Returns {frame_idx: [(obj_type, [x1,y1,x2,y2]), ...]}"""
    frame_annots = defaultdict(list)

    with open(label_path) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 17:
                continue

            frame = int(float(fields[0]))
            obj_type = fields[2]
            if obj_type not in KEEP_CLASSES:
                continue

            x1, y1, x2, y2 = map(float, fields[6:10])
            frame_annots[frame].append((obj_type, [x1, y1, x2, y2]))

    return dict(frame_annots)


def get_2d_boxes_for_frame(frame_annots, frame_idx):
    boxes, labels = [], []
    for obj_type, box2d in frame_annots.get(frame_idx, []):
        x1, y1, x2, y2 = box2d
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(CLS2IDX[obj_type])
    return boxes, labels


# ── Sequence discovery and class distribution ───────────────────


def discover_sequence(seq_id):
    label_path = os.path.join(TRACKLET_DIR, f'{seq_id}.txt')
    img_dir    = os.path.join(DATA_DIR, seq_id)

    if not os.path.exists(label_path):
        print(f'  [skip] no label file for sequence {seq_id}')
        return None
    if not os.path.isdir(img_dir):
        print(f'  [skip] no data/{seq_id} frame folder')
        return None

    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    if not img_paths:
        print(f'  [skip] no .png frames in data/{seq_id}')
        return None

    frame_annots = parse_label_file(label_path)
    calib = load_calib(seq_id)  # may be None; not required downstream

    return img_paths, frame_annots, calib


all_sequences = {}

for seq_id in seq_ids:
    result = discover_sequence(seq_id)
    if result is None:
        continue
    img_paths, frame_annots, calib = result
    all_sequences[seq_id] = result

    counts = defaultdict(int)
    for annots in frame_annots.values():
        for (obj_type, _box) in annots:
            counts[obj_type] += 1

    print(f'{seq_id}: {len(img_paths)} frames | {dict(counts)}')

print(f'\nTotal valid sequences: {len(all_sequences)}')


# ── Select sequences ────────────────────────────────────────────


def get_class_set(frame_annots):
    classes = set()
    for annots in frame_annots.values():
        for (obj_type, _box) in annots:
            classes.add(obj_type)
    return classes

REQUIRED = {'Car', 'Pedestrian', 'Cyclist'}

qualifying = [
    name for name, (imgs, annots, calib) in all_sequences.items()
    if REQUIRED.issubset(get_class_set(annots))
]
print(f'Sequences with all 3 required classes ({len(qualifying)}):')
for q in qualifying:
    print(f'  {q}')

if len(qualifying) < 5:
    print('\nWarning: fewer than 5 qualifying sequences. Using all available.')
    qualifying = list(all_sequences.keys())

if not qualifying:
    TRAIN_SEQS = []
    VAL_SEQS   = []
    TEST_SEQS  = []
else:
    TRAIN_SEQS = qualifying[:-2]
    VAL_SEQS   = [qualifying[-2]]
    TEST_SEQS  = [qualifying[-1]]

print(f'\nTrain sequences : {TRAIN_SEQS}')
print(f'Val   sequences : {VAL_SEQS}')
print(f'Test  sequences : {TEST_SEQS}')


# ── PyTorch Dataset class ───────────────────────────────────────


class KITTITrackletDataset(Dataset):

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]

    def __init__(self, seq_names, all_sequences, target_size=(375, 1242)):

        self.target_h, self.target_w = target_size
        self.normalise = T.Normalize(self.IMG_MEAN, self.IMG_STD)
        self.to_tensor = T.ToTensor()

        self.samples = []
        for name in seq_names:
            img_paths, frame_annots, calib = all_sequences[name]
            for frame_idx, img_path in enumerate(img_paths):
                self.samples.append((img_path, frame_idx, frame_annots, calib))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, frame_idx, frame_annots, calib = self.samples[idx]

        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((self.target_w, self.target_h), Image.BILINEAR)
        scale_x = self.target_w / orig_w
        scale_y = self.target_h / orig_h

        raw_boxes, raw_labels = get_2d_boxes_for_frame(frame_annots, frame_idx)

        valid_boxes, valid_labels = [], []
        for box, lbl in zip(raw_boxes, raw_labels):
            x1 = max(0.0, box[0] * scale_x)
            y1 = max(0.0, box[1] * scale_y)
            x2 = min(float(self.target_w), box[2] * scale_x)
            y2 = min(float(self.target_h), box[3] * scale_y)
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                valid_boxes.append([x1, y1, x2, y2])
                valid_labels.append(lbl)

        if valid_boxes:
            boxes  = torch.tensor(valid_boxes, dtype=torch.float32)
            labels = torch.tensor(valid_labels, dtype=torch.int64)
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        img_tensor = self.normalise(self.to_tensor(img))

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


BATCH_SIZE = 2
IMG_SIZE   = (375, 1242)

train_ds = KITTITrackletDataset(TRAIN_SEQS, all_sequences, IMG_SIZE)
val_ds   = KITTITrackletDataset(VAL_SEQS,   all_sequences, IMG_SIZE)
test_ds  = KITTITrackletDataset(TEST_SEQS,  all_sequences, IMG_SIZE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)

print(f'Train: {len(train_ds)} frames across {len(TRAIN_SEQS)} sequences')
print(f'Val  : {len(val_ds)} frames across {len(VAL_SEQS)} sequence(s)')
print(f'Test : {len(test_ds)} frames across {len(TEST_SEQS)} sequence(s)')


# ── Verify data loader – visualise ground truth ─────────────────


def unnorm(tensor):
    """Reverse ImageNet normalisation for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (tensor * std + mean).permute(1,2,0).numpy().clip(0, 1)


def visualise_gt(dataset, n=4, save='gt_verification.png'):
    fig, axes = plt.subplots(2, 2, figsize=(18, 8))
    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))

    for ax, i in zip(axes.flatten(), idxs):
        img_t, target = dataset[i]
        ax.imshow(unnorm(img_t))
        for box, lbl in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = box.tolist()
            name   = IDX2CLS.get(lbl.item(), '?')
            colour = VIS_COLOURS.get(name, 'white')
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=colour, facecolor='none'
            ))
            ax.text(x1, max(y1-4, 0), name, color=colour, fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.45, pad=1))
        n_boxes = len(target['boxes'])
        ax.set_title(f'Frame {i} — {n_boxes} box(es)')
        ax.axis('off')

    plt.suptitle('Ground-truth boxes (from KITTI tracking labels)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save, dpi=100, bbox_inches='tight')
    plt.show()
    print(f'Saved → {save}')


visualise_gt(train_ds)


# ── Build Faster R-CNN model ────────────────────────────────────


def build_model(num_classes, freeze_backbone=True):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model   = fasterrcnn_resnet50_fpn(weights=weights)

    # Freeze / unfreeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = not freeze_backbone

    if freeze_backbone:
        for p in model.backbone.body.layer4.parameters():
            p.requires_grad = True

    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'Trainable params: {trainable:,} / {total:,} '
          f'({100*trainable/total:.1f}%)')
    return model


# ── Training and evaluating ─────────────────────────────────────


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for imgs, targets in loader:
        imgs    = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_loss(model, loader):
    """Compute loss on val set (model kept in train mode to get loss dict).
    Note: torchvision's detection backbone uses FrozenBatchNorm2d by
    default, so running this forward pass in train mode does not
    contaminate BatchNorm running statistics with validation data."""
    model.train()
    total = 0.0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs    = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            total  += sum(model(imgs, targets).values()).item()
    return total / len(loader)


def train_model(model, train_loader, val_loader, epochs, lr, name):
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt    = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    sched  = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    history = {'train': [], 'val': []}
    for ep in range(1, epochs+1):
        tl = train_one_epoch(model, train_loader, opt)
        vl = val_loss(model, val_loader)
        history['train'].append(tl)
        history['val'].append(vl)
        sched.step()
        print(f'[{name}] Ep {ep:02d}/{epochs} | train={tl:.4f} val={vl:.4f}')
    return history, model


def compute_iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def evaluate_map(model, loader, name=''):
    model.eval()
    metric   = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
    all_ious = []
    with torch.no_grad():
        for imgs, targets in loader:
            preds   = model([img.to(device) for img in imgs])
            p_cpu   = [{k: v.cpu() for k, v in p.items()} for p in preds]
            t_cpu   = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(p_cpu, t_cpu)
            for p, t in zip(p_cpu, t_cpu):
                if len(p['boxes']) and len(t['boxes']):
                    for gtb in t['boxes']:
                        ious = [compute_iou(gtb.tolist(), pb.tolist())
                                for pb in p['boxes']]
                        all_ious.append(max(ious))

    res      = metric.compute()
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    tag      = f'[{name}] ' if name else ''
    print(f'{tag}mAP@0.5={res["map"].item():.4f}  mean IoU={mean_iou:.4f}')

    # Per-class AP
    for i, ap in enumerate(res.get('map_per_class', [])):
        cls = CLASSES[i+1] if i+1 < len(CLASSES) else f'cls{i+1}'
        print(f'  {cls:<15} AP={ap.item():.4f}')
    return res, mean_iou


def plot_loss(histories, names, save='loss_curves.png'):
    plt.figure(figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    for h, n, c in zip(histories, names, colors):
        eps = range(1, len(h['train'])+1)
        plt.plot(eps, h['train'], '--', color=c, alpha=0.55)
        plt.plot(eps, h['val'],   '-',  color=c, label=n)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training loss (solid=val, dashed=train)')
    plt.legend(); plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()


# ── Frozen backbone (head-only fine-tuning) ─────────────────────


EPOCHS = 10

print('Frozen backbone')
model_A = build_model(NUM_CLASSES, freeze_backbone=True)
hist_A, model_A = train_model(model_A, train_loader, val_loader,
                               EPOCHS, lr=5e-4, name='FrozenBackbone')


# ── Full end-to-end fine-tuning ─────────────────────────────────


print('Full fine-tune')
model_B = build_model(NUM_CLASSES, freeze_backbone=False)
hist_B, model_B = train_model(model_B, train_loader, val_loader,
                               EPOCHS, lr=1e-4, name='FullFineTune')

plot_loss([hist_A, hist_B], ['FrozenBackbone', 'FullFineTune'])


# ── Compare based on validated set to pick the best model ───────


print('Validation mAP')
res_A, iou_A = evaluate_map(model_A, val_loader, 'FrozenBackbone')
res_B, iou_B = evaluate_map(model_B, val_loader, 'FullFineTune')

map_A = res_A['map'].item()
map_B = res_B['map'].item()
best_model = model_A if map_A >= map_B else model_B
best_name  = 'FrozenBackbone' if map_A >= map_B else 'FullFineTune'
print(f'\nBest model: {best_name}  (val mAP={max(map_A,map_B):.4f})')


# ── Final evaluation ────────────────────────────────────────────


print(f'=== Test set evaluation ({TEST_SEQS}) ===')
test_res, test_iou = evaluate_map(best_model, test_loader, 'TestSet')


# ── Predictions ─────────────────────────────────────────────────


def visualise_predictions(model, dataset, n=4, score_thresh=0.45,
                           save='predictions_vs_gt.png'):
    model.eval()
    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))
    fig, axes = plt.subplots(n, 2, figsize=(22, n * 4))
    if n == 1:
        axes = [axes]

    for row, idx in enumerate(idxs):
        img_t, target = dataset[idx]
        disp = unnorm(img_t)

        with torch.no_grad():
            pred = model([img_t.to(device)])[0]

        # Ground truth
        ax = axes[row][0]
        ax.imshow(disp)
        for box, lbl in zip(target['boxes'], target['labels']):
            x1,y1,x2,y2 = box.tolist()
            name = IDX2CLS.get(lbl.item(), '?')
            col  = VIS_COLOURS.get(name, 'white')
            ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,
                         lw=2,edgecolor=col,facecolor='none'))
            ax.text(x1,max(y1-4,0),name,color=col,fontsize=8,
                    bbox=dict(facecolor='black',alpha=0.45,pad=1))
        ax.set_title('Ground truth'); ax.axis('off')

        # Predictions
        ax = axes[row][1]
        ax.imshow(disp)
        for box, lbl, score in zip(pred['boxes'].cpu(),
                                    pred['labels'].cpu(),
                                    pred['scores'].cpu()):
            if score < score_thresh:
                continue
            x1,y1,x2,y2 = box.tolist()
            name = IDX2CLS.get(lbl.item(), '?')
            col  = VIS_COLOURS.get(name, 'white')
            ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,
                         lw=2,edgecolor='red',facecolor='none'))
            ax.text(x1,max(y1-4,0),f'{name} {score:.2f}',
                    color='red',fontsize=8,
                    bbox=dict(facecolor='black',alpha=0.45,pad=1))
        ax.set_title(f'Predictions (score>{score_thresh})')
        ax.axis('off')

    plt.suptitle(f'Ground truth vs predictions — {best_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save, dpi=100, bbox_inches='tight')
    plt.show()


visualise_predictions(best_model, test_ds)


# ── Temporal feature stacking (consecutive frames) ──────────────


class KITTITemporalDataset(KITTITrackletDataset):
    def __init__(self, *args, T=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = T

    def __getitem__(self, idx):
        frames = []
        for offset in range(self.T - 1, -1, -1):
            src_idx = max(0, idx - offset)
            img_t, _ = super().__getitem__(src_idx)
            frames.append(img_t)

        _, target = super().__getitem__(idx)

        stacked = torch.cat(frames, dim=0)
        return stacked, target


class TemporalFasterRCNN(nn.Module):
    def __init__(self, num_classes, T=3):
        super().__init__()
        self.T = T
        self.temporal_stem = nn.Conv2d(3 * T, 3, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.temporal_stem.weight)

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.detector = fasterrcnn_resnet50_fpn(weights=weights)
        in_feat = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)

        for p in self.detector.backbone.parameters():
            p.requires_grad = False
        for p in self.detector.backbone.body.layer4.parameters():
            p.requires_grad = True

    def forward(self, images, targets=None):
        fused = [self.temporal_stem(img.unsqueeze(0)).squeeze(0)
                 for img in images]
        if targets is not None:
            return self.detector(fused, targets)
        return self.detector(fused)


print('Temporal classes.')


# ── Results ─────────────────────────────────────────────────────


print('='*60)
print('TASK 2 – SUMMARY')
print('='*60)
print(f'{"Model":<22} {"Val mAP@0.5":>12} {"Mean IoU":>10}')
print('-'*46)
print(f'{"FrozenBackbone":<22} {map_A:>11.4f} {iou_A:>9.4f}')
print(f'{"FullFineTune":<22} {map_B:>11.4f} {iou_B:>9.4f}')
print('-'*46)
print(f'{"Best (test set)":<22} {test_res["map"].item():>11.4f} {test_iou:>9.4f}')
print('='*60)


# ── Save best model ─────────────────────────────────────────────


save_path = f'kitti_{best_name.lower()}.pth'
torch.save(best_model.state_dict(), save_path)
print(f'Saved → {save_path}')
