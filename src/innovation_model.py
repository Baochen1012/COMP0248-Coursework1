import torch
import torch.nn as nn
import torch.nn.functional as F


class InnovationCNN3Head(nn.Module):
    """
    Lightweight innovation on top of baseline:
    1. keep backbone width the same as baseline for saving the computation timing
    2.  segmentation-guided classification
    3. segmentation decoder with skip connections
    4。 bbox head upgraded from single Linear to small MLP
    """
    def __init__(self, in_ch=3, num_classes=10, base=32, guide_alpha=0.3):
        super().__init__()
        self.base = base
        self.guide_alpha = float(guide_alpha)

        # encoder (same width as baseline by default)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # segmentation decoder with skip connections
        self.seg_reduce3 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.seg_fuse2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.seg_fuse1 = nn.Sequential(
            nn.Conv2d(base * 3, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.seg_out = nn.Conv2d(base, 1, 1)

        # segmentation-guided classification
        self.cls_fc = nn.Linear(base * 8, num_classes)

        # slightly stronger bbox head
        self.box_mlp = nn.Sequential(
            nn.Linear(base * 4, base * 2),
            nn.ReLU(inplace=True),
            nn.Linear(base * 2, 4),
        )

    def forward(self, x):
        f1 = self.enc1(x)              # [B, base,   H,   W]
        x = self.down1(f1)
        f2 = self.enc2(x)              # [B, base*2, H/2, W/2]
        x = self.down2(f2)
        f3 = self.enc3(x)              # [B, base*4, H/4, W/4]

        # segmentation decoder with skip fusion
        s = F.interpolate(f3, scale_factor=2, mode='bilinear', align_corners=False)
        s = self.seg_reduce3(s)        # -> [B, base*2, H/2, W/2]
        s = torch.cat([s, f2], dim=1)  # [B, base*4, H/2, W/2]
        s = self.seg_fuse2(s)          # -> [B, base*2, H/2, W/2]

        s = F.interpolate(s, scale_factor=2, mode='bilinear', align_corners=False)
        s = torch.cat([s, f1], dim=1)  # [B, base*3, H, W]
        s = self.seg_fuse1(s)          # -> [B, base, H, W]
        mask_logits = self.seg_out(s)  # [B,1,H,W]

        # classification: raw pooled feature + seg-guided pooled feature
        g_raw = F.adaptive_avg_pool2d(f3, (1, 1)).flatten(1)

        seg_prob = torch.sigmoid(mask_logits)
        seg_attn = F.interpolate(seg_prob, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        f3_guided = f3 * (1.0 + self.guide_alpha * seg_attn)
        g_guided = F.adaptive_avg_pool2d(f3_guided, (1, 1)).flatten(1)

        cls_feat = torch.cat([g_raw, g_guided], dim=1)
        cls_logits = self.cls_fc(cls_feat)

        # bbox still regressed from shared global feature, but with a small MLP
        bbox_pred = torch.sigmoid(self.box_mlp(g_raw))

        return mask_logits, cls_logits, bbox_pred
