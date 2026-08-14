"""
Object detection evaluation: IoU and mAP.

IoU (Intersection over Union):
  Measures overlap between a predicted box and a ground-truth box.
  IoU = area(intersection) / area(union)
  IoU >= 0.5 is the standard threshold for a 'correct' detection.

mAP (Mean Average Precision):
  For each class, compute the area under the precision-recall curve.
  mAP is the mean of these per-class APs.
  Reported at IoU threshold 0.5 (mAP@0.5) following PASCAL VOC convention.
"""

import numpy as np
from typing import Dict, List


def compute_iou(box_pred: np.ndarray, box_gt: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.

    Args:
      box_pred: (4,) predicted box
      box_gt:   (4,) ground-truth box

    Returns:
      iou: float in [0, 1]
    """
    x1 = max(box_pred[0], box_gt[0])
    y1 = max(box_pred[1], box_gt[1])
    x2 = min(box_pred[2], box_gt[2])
    y2 = min(box_pred[3], box_gt[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    area_gt   = (box_gt[2]   - box_gt[0])   * (box_gt[3]   - box_gt[1])
    union = area_pred + area_gt - intersection

    return intersection / union if union > 0 else 0.0


def compute_ap(
    precisions: List[float],
    recalls: List[float],
) -> float:
    """
    Compute Average Precision using the 11-point interpolation method
    (PASCAL VOC convention).
    """
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_thr = [p for p, r in zip(precisions, recalls) if r >= thr]
        ap += max(prec_at_thr) if prec_at_thr else 0.0
    return ap / 11.0


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 3,
) -> Dict[str, float]:
    """
    Compute mAP across all classes.

    Args:
      predictions:   list of dicts with keys 'boxes' (N,4), 'labels' (N,), 'scores' (N,)
      ground_truths: list of dicts with keys 'boxes' (M,4), 'labels' (M,)
      iou_threshold: IoU threshold for a detection to count as TP (default 0.5)
      num_classes:   number of object classes

    Returns:
      dict with per-class AP and overall mAP
    """
    class_names = ['Car', 'Pedestrian', 'Cyclist']
    results = {}
    aps = []

    for cls_idx in range(num_classes):
        tp_list, fp_list, scores_list = [], [], []
        n_gt = 0

        for preds, gts in zip(predictions, ground_truths):
            pred_mask = preds['labels'] == cls_idx
            gt_mask   = gts['labels']  == cls_idx

            pred_boxes = preds['boxes'][pred_mask]
            pred_scores = preds['scores'][pred_mask]
            gt_boxes    = gts['boxes'][gt_mask]

            n_gt += len(gt_boxes)
            matched = set()

            # Sort predictions by confidence score (descending)
            order = np.argsort(-pred_scores)
            for i in order:
                box = pred_boxes[i]
                scores_list.append(pred_scores[i])

                best_iou, best_j = 0.0, -1
                for j, gt_box in enumerate(gt_boxes):
                    if j in matched:
                        continue
                    iou = compute_iou(box, gt_box)
                    if iou > best_iou:
                        best_iou, best_j = iou, j

                if best_iou >= iou_threshold and best_j not in matched:
                    tp_list.append(1)
                    fp_list.append(0)
                    matched.add(best_j)
                else:
                    tp_list.append(0)
                    fp_list.append(1)

        if not scores_list:
            results[class_names[cls_idx]] = 0.0
            aps.append(0.0)
            continue

        order = np.argsort(-np.array(scores_list))
        tp_cumsum = np.cumsum(np.array(tp_list)[order])
        fp_cumsum = np.cumsum(np.array(fp_list)[order])

        recalls    = (tp_cumsum / n_gt).tolist() if n_gt > 0 else [0.0] * len(tp_cumsum)
        precisions = (tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)).tolist()

        ap = compute_ap(precisions, recalls)
        results[class_names[cls_idx]] = ap
        aps.append(ap)

    results['mAP'] = float(np.mean(aps))
    return results
