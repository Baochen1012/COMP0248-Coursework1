import os
import json
import numpy as np
import torch

def lp(p):
    p = os.path.abspath(str(p))
    if os.name != "nt":
        return p
    if p.startswith('\\?\\'):
        return p
    if p.startswith('\\\\'):
        return '\\?\\UNC\\' + p.lstrip('\\')
    return '\\?\\' + p

def bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    uni = aa + ab - inter
    return 0.0 if uni <= 0 else inter / uni

def mask_iou(pred01, gt01):
    p = pred01.detach().cpu().numpy() if isinstance(pred01, torch.Tensor) else pred01
    g = gt01.detach().cpu().numpy() if isinstance(gt01, torch.Tensor) else gt01
    p = (p > 0.5).astype(np.uint8); g = (g > 0.5).astype(np.uint8)
    inter = (p & g).sum(); uni = (p | g).sum()
    return 0.0 if uni == 0 else float(inter) / float(uni)

def dice_score(pred01, gt01, eps=1e-6):
    p = pred01.detach().cpu().numpy() if isinstance(pred01, torch.Tensor) else pred01
    g = gt01.detach().cpu().numpy() if isinstance(gt01, torch.Tensor) else gt01
    p = (p > 0.5).astype(np.uint8); g = (g > 0.5).astype(np.uint8)
    inter = (p & g).sum(); s = p.sum() + g.sum()
    return float((2.0 * inter + eps) / (s + eps))

def mask_to_bbox_xyxy(mask01):
    m = mask01.detach().cpu().numpy() if isinstance(mask01, torch.Tensor) else mask01
    m = (m > 0.5).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return (0, 0, 1, 1)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return (x1, y1, x2, y2)

def det_acc_at_iou(pred_bbox, gt_bbox, thr=0.5):
    return 1.0 if bbox_iou_xyxy(pred_bbox, gt_bbox) >= thr else 0.0

def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def macro_f1_from_cm(cm, eps=1e-9):
    C = cm.shape[0]
    f1s = []
    for c in range(C):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1s.append(2 * prec * rec / (prec + rec + eps))
    return float(np.mean(f1s))

def top1_acc(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return 0.0 if y_true.size == 0 else float((y_true == y_pred).mean())

def save_json(obj, path):
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
