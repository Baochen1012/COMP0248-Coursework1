import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import DataLoader

from dataloader import build_train_val, build_test
from model import SimpleCNN3Head
import utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=str, default='baseline_rgb')
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--use_depth', action='store_true')
    p.add_argument('--depth_mode', type=str, default='depth', choices=['depth','depth_raw'])
    p.add_argument('--split', type=str, default='val', choices=['val','test'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--num', type=int, default=40)
    p.add_argument('--batch', type=int, default=8)
    return p.parse_args()


def blend_mask(img_rgb, mask01, color, alpha=0.35):
    overlay = Image.new('RGB', img_rgb.size, color)
    m = (mask01 * 255).astype(np.uint8)
    m_img = Image.fromarray(m, mode='L')
    out = Image.composite(overlay, img_rgb, m_img)
    return Image.blend(img_rgb, out, alpha)


@torch.no_grad()
def main():
    args = parse_args()
    pr = Path(__file__).resolve().parents[1]
    collated = pr / 'dataset' / 'RGB_depth_annotations'
    test_root = pr / 'dataset' / 'Test data' / 'COMP0248_Test_data_23'

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = pr / 'weights' / f'{args.run}_best.pth'
    ckpt = Path(ckpt)

    in_ch = 4 if args.use_depth else 3
    model = SimpleCNN3Head(in_ch=in_ch, num_classes=10, base=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    data = torch.load(str(ckpt), map_location='cpu')
    model.load_state_dict(data['model'], strict=True)
    model.eval()

    if args.split == 'val':
        _, ds, _ = build_train_val(collated_root=collated, val_ratio=args.val_ratio,
                                   seed=args.seed, use_depth=args.use_depth,
                                   depth_mode=args.depth_mode, strict=True)
    else:
        ds, _ = build_test(test_root=test_root, use_depth=args.use_depth,
                           depth_mode=args.depth_mode, strict=True)

    ds.return_meta = False
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    out_dir = pr / 'results' / args.run / 'overlays' / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        gt_mask = batch['mask'].cpu().numpy()
        gt_bbox = batch['bbox'].cpu().numpy()
        gt_label = batch['label'].cpu().numpy()

        mask_logits, cls_logits, bbox_pred = model(img)

        pred_prob = torch.sigmoid(mask_logits).cpu().numpy()
        pred_mask = (pred_prob > 0.5).astype(np.uint8)
        pred_label = torch.argmax(cls_logits, dim=1).cpu().numpy()

        B, _, H, W = batch['mask'].shape
        pred_bbox_px = bbox_pred.detach().cpu().numpy()
        pred_bbox_px[:, [0, 2]] *= float(W)
        pred_bbox_px[:, [1, 3]] *= float(H)

        img_rgb = (img[:, :3].detach().cpu().numpy() * 255.0).astype(np.uint8)

        B = img_rgb.shape[0]
        for i in range(B):
            if saved >= args.num:
                print('saved overlays:', saved)
                return

            arr = np.transpose(img_rgb[i], (1,2,0))
            pil = Image.fromarray(arr, mode='RGB')

            gm = gt_mask[i,0].astype(np.uint8)
            pm = pred_mask[i,0].astype(np.uint8)

            pil = blend_mask(pil, gm, (0,255,0), 0.30)   # GT green
            pil = blend_mask(pil, pm, (255,0,0), 0.30)   # Pred red

            draw = ImageDraw.Draw(pil)

            gb = tuple(gt_bbox[i].tolist())
            pb = tuple(pred_bbox_px[i].tolist())

            draw.rectangle([gb[0], gb[1], gb[2]-1, gb[3]-1], outline=(0,255,0), width=2)
            draw.rectangle([pb[0], pb[1], pb[2]-1, pb[3]-1], outline=(255,0,0), width=2)

            iou = utils.bbox_iou_xyxy(pb, gb)
            di = utils.dice_score(pm, gm)

            txt = f'gt={int(gt_label[i])} pred={int(pred_label[i])} iou={iou:.2f} dice={di:.2f}'
            draw.text((5,5), txt, fill=(255,255,255))

            pil.save(str(out_dir / f'{saved:04d}.png'))
            saved += 1

    print('saved overlays:', saved)


if __name__ == '__main__':
    main()
