import argparse
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from dataloader import build_train_val, build_test
from innovation_model import InnovationCNN3Head
import utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=str, default='innovation_rgb_light')
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--base', type=int, default=32)
    p.add_argument('--guide_alpha', type=float, default=0.3)
    return p.parse_args()


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    det_accs, bbox_ious, seg_ious, dices = [], [], [], []
    y_true, y_pred = [], []

    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        gt_mask = batch['mask'].to(device, non_blocking=True)
        gt_bbox_px = batch['bbox'].to(device, non_blocking=True).float()
        gt_label = batch['label'].to(device, non_blocking=True)

        B, _, H, W = gt_mask.shape
        mask_logits, cls_logits, bbox_pred = model(img)

        pred_prob = torch.sigmoid(mask_logits)
        pred_mask = (pred_prob > 0.5).float()
        pred_label = torch.argmax(cls_logits, dim=1).detach().cpu().numpy()

        pred_bbox_px = bbox_pred.detach().clone()
        pred_bbox_px[:, [0, 2]] *= float(W)
        pred_bbox_px[:, [1, 3]] *= float(H)

        gt_bbox_np = gt_bbox_px.detach().cpu().numpy()
        pred_bbox_np = pred_bbox_px.detach().cpu().numpy()

        for i in range(B):
            pm = pred_mask[i, 0]
            gm = gt_mask[i, 0]
            seg_ious.append(utils.mask_iou(pm, gm))
            dices.append(utils.dice_score(pm, gm))

            pb = tuple(pred_bbox_np[i].tolist())
            gb = tuple(gt_bbox_np[i].tolist())
            bbox_ious.append(utils.bbox_iou_xyxy(pb, gb))
            det_accs.append(utils.det_acc_at_iou(pb, gb, thr=0.5))

        y_true.extend(gt_label.detach().cpu().numpy().tolist())
        y_pred.extend(pred_label.tolist())

    cm = utils.confusion_matrix(y_true, y_pred, num_classes=10)
    out = {
        'det_acc@0.5': float(sum(det_accs) / max(1, len(det_accs))),
        'mean_bbox_iou': float(sum(bbox_ious) / max(1, len(bbox_ious))),
        'mean_seg_iou': float(sum(seg_ious) / max(1, len(seg_ious))),
        'mean_dice': float(sum(dices) / max(1, len(dices))),
        'top1_acc': utils.top1_acc(y_true, y_pred),
        'macro_f1': utils.macro_f1_from_cm(cm),
        'confusion_matrix': cm.tolist(),
        'num_samples': int(len(y_true)),
    }
    return out


def main():
    args = parse_args()
    pr = Path(__file__).resolve().parents[1]
    collated = pr / 'dataset' / 'RGB_depth_annotations'
    test_root = pr / 'dataset' / 'Test data' / 'COMP0248_Test_data_23'

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = pr / 'weights' / f'{args.run}_best.pth'
    ckpt = Path(ckpt)

    model = InnovationCNN3Head(in_ch=3, num_classes=10, base=args.base, guide_alpha=args.guide_alpha)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    data = torch.load(str(ckpt), map_location='cpu')
    model.load_state_dict(data['model'], strict=True)

    if args.split == 'val':
        _, ds, _ = build_train_val(
            collated_root=collated,
            val_ratio=args.val_ratio,
            seed=args.seed,
            use_depth=False,
            depth_mode='depth',
            strict=True,
        )
        out_path = pr / 'results' / args.run / 'metrics_val.json'
    else:
        ds, _ = build_test(test_root=test_root, use_depth=False, depth_mode='depth', strict=True)
        out_path = pr / 'results' / args.run / 'metrics_test.json'

    ds.return_meta = False
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    metrics = run_eval(model, loader, device)
    utils.save_json(metrics, out_path)

    print('saved:', out_path)
    print(json.dumps(metrics, indent=2)[:800], '...')


if __name__ == '__main__':
    main()
