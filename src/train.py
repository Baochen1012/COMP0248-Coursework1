import time
import argparse
from pathlib import Path


import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataloader import build_train_val
from model import SimpleCNN3Head
import utils
from collections import OrderedDict

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--use_depth', action='store_true')
    p.add_argument('--depth_mode', type=str, default='depth', choices=['depth','depth_raw'])
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--run', type=str, default='baseline_rgb')
    p.add_argument('--num_workers', type=int, default=4)

    # loss weights
    p.add_argument('--w_seg', type=float, default=1.0)
    p.add_argument('--w_cls', type=float, default=1.0)
    p.add_argument('--w_det', type=float, default=1.0)

    # early stop
    p.add_argument('--patience', type=int, default=8)       # 几轮不变就停 / stop if no improve
    p.add_argument('--min_delta', type=float, default=1e-4) # 下降小于这个算没改善

    return p.parse_args()


def fmt(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)


def print_epoch(epoch, epochs, train_loss, val_loss, valm, best_val_loss):
    msg = (
        f"\n[{epoch:03d}/{epochs:03d}] "
        f"train_loss {fmt(train_loss, 4)}  val_loss {fmt(val_loss, 4)} | "
        f"segIoU {fmt(valm['mean_seg_iou'], 3)} dice {fmt(valm['mean_dice'], 3)} | "
        f"detAcc {fmt(valm['det_acc@0.5'], 3)} bboxIoU {fmt(valm['mean_bbox_iou'], 3)} | "
        f"clsAcc {fmt(valm['top1_acc'], 3)} F1 {fmt(valm['macro_f1'], 3)} | "
        f"bestVal_loss {fmt(best_val_loss, 4)}"
    )
    print(msg)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _bbox_norm_xyxy_from_px(gt_bbox_px, H, W):
    b = gt_bbox_px.clone()
    b[:, [0, 2]] /= float(W)
    b[:, [1, 3]] /= float(H)
    return b.clamp_(0.0, 1.0)


@torch.no_grad()
def quick_val(model, loader, device, loss_fns, weights):
    bce, ce, l1 = loss_fns
    w_seg, w_cls, w_det = weights

    model.eval()

    det_accs, bbox_ious, seg_ious, dices = [], [], [], []
    y_true, y_pred = [], []

    tot_loss = 0.0
    tot_n = 0

    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        gt_mask = batch['mask'].to(device, non_blocking=True)
        gt_bbox_px = batch['bbox'].to(device, non_blocking=True).float()
        gt_label = batch['label'].to(device, non_blocking=True)

        B, _, H, W = gt_mask.shape
        gt_bbox_norm = _bbox_norm_xyxy_from_px(gt_bbox_px, H, W)

        mask_logits, cls_logits, bbox_pred = model(img)

        loss_seg = bce(mask_logits, gt_mask)
        loss_cls = ce(cls_logits, gt_label)
        loss_det = l1(bbox_pred, gt_bbox_norm)

        loss = w_seg*loss_seg + w_cls*loss_cls + w_det*loss_det
        tot_loss += float(loss.item()) * B
        tot_n += B

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

    metrics = {
        'det_acc@0.5': float(sum(det_accs) / max(1, len(det_accs))),
        'mean_bbox_iou': float(sum(bbox_ious) / max(1, len(bbox_ious))),
        'mean_seg_iou': float(sum(seg_ious) / max(1, len(seg_ious))),
        'mean_dice': float(sum(dices) / max(1, len(dices))),
        'top1_acc': utils.top1_acc(y_true, y_pred),
        'macro_f1': utils.macro_f1_from_cm(cm),
    }
    val_loss = tot_loss / max(1, tot_n)
    return val_loss, metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    pr = Path(__file__).resolve().parents[1]
    collated = pr / 'dataset' / 'RGB_depth_annotations'

    train_ds, val_ds, _ = build_train_val(
        collated_root=collated,
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_depth=args.use_depth,
        depth_mode=args.depth_mode,
        strict=True,
        save_split_to=pr / 'results' / 'splits' / f'split_seed{args.seed}.json',
    )
    train_ds.return_meta = False
    val_ds.return_meta = False

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    in_ch = 4 if args.use_depth else 3
    model = SimpleCNN3Head(in_ch=in_ch, num_classes=10, base=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    l1 = nn.SmoothL1Loss()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir = pr / 'results' / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = pr / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_val = 1e18
    bad_epochs = 0
    log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0

        pbar = tqdm(train_loader, desc=f"ep {epoch:03d}", leave=False)

        for batch in pbar:
            img = batch['image'].to(device, non_blocking=True)
            gt_mask = batch['mask'].to(device, non_blocking=True)
            gt_bbox_px = batch['bbox'].to(device, non_blocking=True).float()
            gt_label = batch['label'].to(device, non_blocking=True)

            B, _, H, W = gt_mask.shape
            gt_bbox_norm = _bbox_norm_xyxy_from_px(gt_bbox_px, H, W)

            opt.zero_grad(set_to_none=True)
            mask_logits, cls_logits, bbox_pred = model(img)

            loss_seg = bce(mask_logits, gt_mask)
            loss_cls = ce(cls_logits, gt_label)
            loss_det = l1(bbox_pred, gt_bbox_norm)

            loss = args.w_seg*loss_seg + args.w_cls*loss_cls + args.w_det*loss_det

            loss.backward()
            opt.step()

            pbar.set_postfix(OrderedDict([
                ("overall_loss", f"{loss.item():.4f}"),
                ("seg_loss", f"{loss_seg.item():.4f}"),
                ("cls_loss", f"{loss_cls.item():.4f}"),
                ("det_loss", f"{loss_det.item():.4f}"),
            ]))

            tot += float(loss.item()) * B
            n += B

        train_loss = tot / max(1, n)

        val_loss, valm = quick_val(
            model, val_loader, device,
            loss_fns=(bce, ce, l1),
            weights=(args.w_seg, args.w_cls, args.w_det),
        )

        improved = (best_val - val_loss) > args.min_delta
        if improved:
            best_val = val_loss
            bad_epochs = 0
            torch.save({'model': model.state_dict(), 'args': vars(args), 'best_val': best_val, 'epoch': epoch},
                       weights_dir / f'{args.run}_best.pth')
        else:
            bad_epochs += 1

        torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': epoch},
                   weights_dir / f'{args.run}_last.pth')

        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'time_sec': round(time.time()-t0, 2),
            **valm,
            'best_val': best_val,
            'bad_epochs': bad_epochs,
        }
        log.append(row)
        utils.save_json(log, run_dir / 'train_log.json')

        print_epoch(epoch, args.epochs, train_loss, val_loss, valm, best_val)

        if bad_epochs >= args.patience:
            print(f"early stop: val_loss not improved for {args.patience} epochs")
            break

    print('done')


if __name__ == '__main__':
    main()
